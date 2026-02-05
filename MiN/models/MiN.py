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

    def _map_targets(self, targets, data_manger):
        """Helper: Map targets sang order 0..N ngay trong loop để tránh lỗi Index/CUDA"""
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        mapped = [data_manger.map_cat2order(t) for t in targets]
        return torch.tensor(mapped, device=self.device, dtype=torch.long)

    def calculate_drift(self, clean_loader, data_manger):
        """Tính Drift Matrix P (TSSP)"""
        self.logger.info("Calculating Drift Matrix P (TSSP)...")
        self._network.eval()
        self._old_network.eval()
        
        XtX = torch.zeros(self.buffer_size, self.buffer_size, device='cpu')
        XtY = torch.zeros(self.buffer_size, self.buffer_size, device='cpu')
        
        with torch.no_grad():
            for _, inputs, targets in clean_loader:
                inputs = inputs.to(self.device)
                # Map targets (cho đồng bộ)
                targets = self._map_targets(targets, data_manger)
                
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

    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        
        # [FIX]: Đồng bộ labels
        mapped_labels = self.cat2order(test_set.labels, data_manger)
        test_set.labels = mapped_labels
        if hasattr(test_set, 'targets'): test_set.targets = mapped_labels

        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader, data_manger)
        
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def init_train(self, data_manger):
        self.cur_task += 1
        self.known_class = self.init_class
        
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info(f"task_list: {train_list_name}")
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
             for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        # 1. Run (Train Noise)
        self.run(train_loader, data_manger)
        self._network.collect_projections(mode='threshold', val=0.95)
        # [REMOVED]: self._network.after_task_magmax_merge()
        self._clear_gpu()
        
        # 2. Final Fit (Gom stats)
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, test_loader, data_manger, init_mode=False)

        # 3. Refit (Task 0 Update)
        self.re_fit(fit_loader, test_loader, data_manger)
        
        del train_set, test_set
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        self.known_class += self.increment
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"task_list: {train_list_name}")
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
             for param in self._network.backbone.parameters(): param.requires_grad = False

        # 1. Init Fit
        self.fit_fc(train_loader, test_loader, data_manger, init_mode=True)

        self._network.update_fc(self.increment)
        self._network.update_noise()
        
        # 2. Run
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.run(run_loader, data_manger)
        self._network.collect_projections(mode='threshold', val=0.95)
        # [REMOVED]: self._network.after_task_magmax_merge()
        self._clear_gpu()
        
        # 3. Final Fit
        del train_set
        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        self.fit_fc(train_loader, test_loader, data_manger, init_mode=False)

        # 4. Refit
        self.re_fit(train_loader, test_loader, data_manger)
        
        del train_set, test_set
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader, data_manger, init_mode=False):
        self._network.eval()
        self._network.to(self.device)
        
        if init_mode:
            # Init Fit: Gom 1 lần (dùng num_workers của loader chính)
            for _ in tqdm(range(self.fit_epoch), desc="Fit FC (Init)"):
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(self.device)
                    targets = self._map_targets(targets, data_manger)
                    targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
                    self._network.accumulate_stats(inputs, targets)
            
            self._network.fit(init_mode=True)
            self._clear_gpu()
            
        else:
            # Final Fit: Streaming từng class
            # [QUAN TRỌNG]: Lấy nhãn RAW từ dataset để lọc (vì Subset trỏ vào dataset gốc)
            dataset = train_loader.dataset
            if hasattr(dataset, 'labels'): raw_classes = sorted(list(set(dataset.labels)))
            elif hasattr(dataset, 'targets'): raw_classes = sorted(list(set(dataset.targets)))
            else: raw_classes = sorted(list(set([x[1] for x in dataset])))

            print(f"--> [Streaming Fit] Processing {len(raw_classes)} classes...")
            
            for raw_c in raw_classes:
                if hasattr(dataset, 'labels'): indices = np.where(np.array(dataset.labels) == raw_c)[0]
                elif hasattr(dataset, 'targets'): indices = np.where(np.array(dataset.targets) == raw_c)[0]
                else: indices = [i for i, x in enumerate(dataset) if x[1] == raw_c]
                
                if len(indices) == 0: continue

                # [FIX TREO MÁY]: Ép num_workers=0 cho DataLoader con
                # Khi chạy loop tạo nhiều DataLoader nhỏ liên tục, nếu có worker > 0 sẽ gây deadlock
                sub_loader = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch, shuffle=False, num_workers=0)
                
                for _ in range(self.fit_epoch):
                    for _, inputs, targets in sub_loader:
                        inputs = inputs.to(self.device)
                        targets = self._map_targets(targets, data_manger)
                        targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class + (self.increment if self.cur_task > 0 else 0))
                        self._network.accumulate_stats(inputs, targets)
                
                self._network.compress_stats()
                self._clear_gpu()

            self._network.fit(init_mode=False)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader, data_manger):
        self._network.eval()
        self._network.to(self.device)
        
        # 1. Tính Drift
        P_drift = None
        boundary = 0
        if self.cur_task > 0:
            P_drift = self.calculate_drift(train_loader, data_manger)
            boundary = self.known_class

        # 2. Update Weight
        self.logger.info("--> [DPCR] Applying Drift Correction...")
        self._network.fit(P_drift=P_drift, known_classes_boundary=boundary, init_mode=False)
        self._clear_gpu()

    def run(self, train_loader, data_manger):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
        
        # [FIX SCALE]: Hardcode 0.85 theo yêu cầu
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
                inputs = inputs.to(self.device)
                targets = self._map_targets(targets, data_manger)
                
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

    def eval_task(self, test_loader, data_manger):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = self._map_targets(targets, data_manger)
                
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
        if isinstance(targets, np.ndarray): targets = targets.tolist()
        elif isinstance(targets, torch.Tensor): targets = targets.cpu().numpy().tolist()
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets