import math
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import copy
import gc 
import os

from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler

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
        
        XtX = torch.zeros(self.args['buffer_size'], self.args['buffer_size'], device=self.device)
        XtY = torch.zeros(self.args['buffer_size'], self.args['buffer_size'], device=self.device)
        
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                inputs = inputs.to(self.device)
                with autocast('cuda', enabled=False):
                    f_old = self._old_network.buffer(self._old_network.backbone(inputs).float())
                    f_new = self._network.buffer(self._network.backbone(inputs).float())
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        reg = 1e-4 * torch.eye(self.args['buffer_size'], device=self.device)
        try:
            P = torch.linalg.solve(XtX + reg, XtY)
        except:
            P = torch.inverse(XtX + reg) @ XtY
        return P

    # --- TASK 0 ---
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
        
        # 1. Run 
        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._network.after_task_magmax_merge()
        
        # 2. Fit Final
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, init_mode=False)

        # 3. Refit
        self.re_fit(None, None) 
        
        self.after_train(data_manger)

    # --- TASK > 0 ---
    def increment_train(self, data_manger):
        self.cur_task += 1
        self.known_class += self.increment
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"task_list: {train_list_name}")
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        self._network.update_fc(self.increment)
        self._network.update_noise()

        # 1. Fit Init
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, init_mode=True)

        # 2. Run
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._network.after_task_magmax_merge()
        
        # 3. Fit Final
        self.fit_fc(fit_loader, init_mode=False)

        # 4. Refit
        clean_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        clean_set.labels = self.cat2order(clean_set.labels, data_manger)
        clean_loader = DataLoader(clean_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.re_fit(clean_loader, None)
        
        self.after_train(data_manger)

    def fit_fc(self, train_loader, test_loader=None, init_mode=False):
        self._network.eval()
        self._network.to(self.device)
        
        if init_mode:
            desc = f"Task {self.cur_task} Fit (Init - RLS)"
            for _ in tqdm(range(self.fit_epoch), desc=desc):
                for _, inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    y_onehot = F.one_hot(targets, num_classes=self._network.known_class)
                    self._network.fit(inputs, y_onehot)
            self._network.solve_temporary_analytic()
            
        else:
            # Streaming Mode
            dataset = train_loader.dataset
            start_class = self.known_class - self.increment if self.cur_task > 0 else 0
            end_class = self.known_class
            
            print(f"--> [GPU Stream Fit] Processing classes {start_class} to {end_class} sequentially...")
            
            for c in range(start_class, end_class):
                if hasattr(dataset, 'labels'): 
                    indices = np.where(np.array(dataset.labels) == c)[0]
                elif hasattr(dataset, 'targets'):
                    indices = np.where(np.array(dataset.targets) == c)[0]
                else:
                    indices = [i for i, x in enumerate(dataset) if x[1] == c]
                
                if len(indices) == 0: continue

                sub_set = Subset(dataset, indices)
                sub_loader = DataLoader(sub_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
                
                for _ in range(self.fit_epoch):
                    for _, inputs, targets in sub_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        y_onehot = F.one_hot(targets, num_classes=self._network.known_class)
                        self._network.fit(inputs, y_onehot)
                
                self._network.finalize_task_stats() 
                self._clear_gpu()
                print(f"    -> Class {c}: Compressed & VRAM cleared.")

        self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        if self.cur_task > 0:
            self.logger.info("--> [DPCR] Estimating Shift & Reconstructing...")
            P_drift = self.calculate_drift(train_loader)
            old_class_boundary = self._network.known_class - self.increment
            self._network.reconstruct_classifier_dpcr(P_drift, known_classes_boundary=old_class_boundary)
        else:
            self._network.reconstruct_classifier_dpcr(None, known_classes_boundary=0)
        self._clear_gpu()

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

        current_scale = 0.85 
        if self.cur_task > 0:
            current_scale = self.compute_adaptive_scale(train_loader)

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
                        
                        # [FIX LOGIC]: Padding logits1 để khớp size với logits2
                        old_dim = logits1.shape[1]
                        new_dim = logits2.shape[1]
                        if new_dim > old_dim:
                            padding = torch.full((logits1.shape[0], new_dim - old_dim), float('-inf'), device=self.device)
                            logits1 = torch.cat([logits1, padding], dim=1)
                        
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                
                if self.cur_task > 0:
                    if epoch >= WARMUP_EPOCHS:
                        self.scaler.unscale_(optimizer)
                        self._network.apply_gpm_to_grads(scale=current_scale)

                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        
        self.logger.info(info)

    def compute_adaptive_scale(self, current_loader):
        curr_proto = self.get_task_prototype(self._network, current_loader)
        if not hasattr(self, 'old_prototypes'): self.old_prototypes = []
        if not self.old_prototypes:
            self.old_prototypes.append(curr_proto)
            return 0.95
        max_sim = 0.0
        curr_norm = F.normalize(curr_proto.unsqueeze(0), p=2, dim=1)
        for old_p in self.old_prototypes:
            old_norm = F.normalize(old_p.unsqueeze(0), p=2, dim=1)
            sim = torch.mm(curr_norm, old_norm.t()).item()
            if sim > max_sim: max_sim = sim
        self.old_prototypes.append(curr_proto)
        scale = 0.5 + 0.5 * (1.0 - max_sim)
        return max(0.65, min(scale, 0.95))

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
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
    
    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

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

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
                label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }
    
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
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        print('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        del test_set