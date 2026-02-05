import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import copy
import gc 
import os

from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# Import Mixed Precision
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        self.scaler = GradScaler('cuda')
        self._old_network = None 

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calculate_drift(self, clean_loader):
        self.logger.info("Calculating Drift Matrix P (TSSP)...")
        self._network.eval()
        self._old_network.eval()
        
        XtX = torch.zeros(self.buffer_size, self.buffer_size, device='cpu')
        XtY = torch.zeros(self.buffer_size, self.buffer_size, device='cpu')
        
        with torch.no_grad():
            for _, inputs, targets in clean_loader:
                inputs = inputs.to(self.device)
                with autocast('cuda', enabled=False):
                    f_old = self._old_network.buffer(self._old_network.backbone(inputs).float()).cpu()
                    f_new = self._network.buffer(self._network.backbone(inputs).float()).cpu()
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        reg = 1e-4 * torch.eye(self.buffer_size, device='cpu')
        try:
            P = torch.linalg.solve(XtX + reg, XtY)
        except:
            P = torch.inverse(XtX + reg) @ XtY
        return P.to(self.device)

    def init_train(self, data_manger):
        self.cur_task += 1
        self.known_class = self.init_class
        
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info(f"task_list: {train_list_name}")
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        if self.args['pretrained']:
             for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        # 1. Run
        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # 2. Fit Final
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, init_mode=False)

        # 3. Refit
        self.re_fit(None, None) 
        
        self.after_train(data_manger)

    def increment_train(self, data_manger):
        self.cur_task += 1
        self.known_class += self.increment
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"task_list: {train_list_name}")
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        # Update FC ngay đầu
        self._network.update_fc(self.increment)
        self._network.update_noise()

        # 1. Fit Init
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, init_mode=True)

        # 2. Run
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        del train_set

        # 3. Fit Final
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        self.fit_fc(train_loader, init_mode=False)

        # 4. Refit
        self.re_fit(train_loader)
        
        del train_set
        self._clear_gpu()
        self.after_train(data_manger)

    def fit_fc(self, train_loader, init_mode=False):
        self._network.eval()
        self._network.to(self.device)
        
        if init_mode:
            # Init Fit: Gom 1 lần
            for _ in tqdm(range(self.fit_epoch), desc="Fit FC (Init)"):
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
                    self._network.accumulate_stats(inputs, targets)
            
            self._network.fit(init_mode=True)
            self._clear_gpu()
            
        else:
            # Final Fit: Streaming từng class
            dataset = train_loader.dataset
            if hasattr(dataset, 'labels'): classes = sorted(list(set(dataset.labels)))
            elif hasattr(dataset, 'targets'): classes = sorted(list(set(dataset.targets)))
            else: classes = sorted(list(set([x[1] for x in dataset])))

            print(f"--> [Streaming Fit] Processing {len(classes)} classes...")
            
            for c in classes:
                if hasattr(dataset, 'labels'): indices = np.where(np.array(dataset.labels) == c)[0]
                elif hasattr(dataset, 'targets'): indices = np.where(np.array(dataset.targets) == c)[0]
                else: indices = [i for i, x in enumerate(dataset) if x[1] == c]
                
                if len(indices) == 0: continue

                # [FIX HANG]: num_workers=0
                sub_loader = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch, shuffle=False, num_workers=0)
                
                for _ in range(self.fit_epoch):
                    for _, inputs, targets in sub_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
                        self._network.accumulate_stats(inputs, targets)
                
                self._network.compress_stats()
                # [FIX OOM]: Xóa cache ngay sau mỗi class
                self._clear_gpu()

            self._network.fit(init_mode=False)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader=None):
        self._network.eval()
        self._network.to(self.device)
        
        P_drift = None
        boundary = 0
        if self.cur_task > 0:
            P_drift = self.calculate_drift(train_loader)
            boundary = self.known_class - self.increment

        self.logger.info("--> [DPCR] Drift Correction...")
        self._network.fit(P_drift=P_drift, known_classes_boundary=boundary, init_mode=False)
        self._clear_gpu()

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
        current_scale = 0.85 

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
        
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        WARMUP_EPOCHS = 2

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 

                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        
                        if logits2.shape[1] > logits1.shape[1]:
                            padding = torch.zeros((logits1.shape[0], logits2.shape[1] - logits1.shape[1]), device=self.device)
                            logits1 = torch.cat([logits1, padding], dim=1)
                        
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                
                if self.cur_task > 0 and epoch >= WARMUP_EPOCHS:
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=current_scale)

                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            if epoch % 5 == 0: self._clear_gpu()
        self.logger.info(info)

    def compute_adaptive_scale(self, current_loader): return 0.85

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                features.append(feature.detach().cpu())
        all_features = torch.cat(features, dim=0)
        prototype = torch.mean(all_features, dim=0).to(self.device)
        self._clear_gpu()
        return prototype

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        device = self.device
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def after_train(self, data_manger):
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        del test_set