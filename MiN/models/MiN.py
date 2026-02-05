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
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler

# Mixed Precision
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        # [CƯỠNG CHẾ BỘ NHỚ]
        if args['buffer_size'] > 4096:
            print(f"!!! Force buffer_size {args['buffer_size']} -> 2048 to prevent OOM")
            args['buffer_size'] = 2048

        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = 0 # Ép về 0 để tránh treo luồng

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
        self.buffer_batch = args["buffer_batch"]
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        self.scaler = GradScaler('cuda')
        self._old_network = None 

    def _clear_gpu(self):
        gc.collect()
        torch.cuda.empty_cache()

    def calculate_drift(self, clean_loader):
        self._network.eval()
        self._old_network.eval()
        bs = self.args['buffer_size']
        XtX = torch.zeros(bs, bs, device='cpu')
        XtY = torch.zeros(bs, bs, device='cpu')
        
        with torch.no_grad():
            for _, inputs, targets in clean_loader:
                inputs = inputs.to(self.device)
                f_old = self._old_network.buffer(self._old_network.backbone(inputs).float()).cpu()
                f_new = self._network.buffer(self._network.backbone(inputs).float()).cpu()
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        reg = 1e-4 * torch.eye(bs, device='cpu')
        P = torch.linalg.solve(XtX + reg, XtY)
        return P.to(self.device)

    def init_train(self, data_manger):
        self.cur_task += 1
        self.known_class = self.init_class
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=0)
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0)
        self.fit_fc(fit_loader, init_mode=False)
        self.re_fit(None)
        self.after_train(data_manger)

    def increment_train(self, data_manger):
        self.cur_task += 1
        self.known_class += self.increment
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        
        train_list, _, _ = data_manger.get_task_list(self.cur_task)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        self._network.update_fc(self.increment)
        self._network.update_noise()

        fit_init_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0)
        self.fit_fc(fit_init_loader, init_mode=True)

        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        
        train_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_no_aug.labels = self.cat2order(train_no_aug.labels, data_manger)
        fit_final_loader = DataLoader(train_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=0)
        
        self.fit_fc(fit_final_loader, init_mode=False)
        self.re_fit(fit_final_loader)
        self.after_train(data_manger)

    def fit_fc(self, train_loader, init_mode=False):
        self._network.eval()
        if init_mode:
            for _ in range(self.fit_epoch):
                for _, inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    y_onehot = F.one_hot(targets, num_classes=self._network.known_class)
                    self._network.accumulate_stats(inputs, y_onehot)
            self._network.fit(init_mode=True)
        else:
            dataset = train_loader.dataset
            classes = sorted(list(set(dataset.labels)))
            for c in classes:
                indices = np.where(np.array(dataset.labels) == c)[0]
                sub_loader = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch, shuffle=False, num_workers=0)
                for _ in range(self.fit_epoch):
                    for _, inputs, targets in sub_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        y_onehot = F.one_hot(targets, num_classes=self._network.known_class)
                        self._network.accumulate_stats(inputs, y_onehot)
                self._network.compress_stats()
                self._clear_gpu()
            self._network.fit(init_mode=False)
        self._clear_gpu()

    def re_fit(self, train_loader):
        P_drift, boundary = None, 0
        if self.cur_task > 0:
            P_drift = self.calculate_drift(train_loader)
            boundary = self.known_class - self.increment
        self._network.fit(P_drift=P_drift, known_classes_boundary=boundary, init_mode=False)
        self._clear_gpu()
    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
        current_scale = 0.85 

        # 1. Cấu hình đóng băng/giải băng tham số
        for param in self._network.parameters(): 
            param.requires_grad = False
        for param in self._network.normal_fc.parameters(): 
            param.requires_grad = True
        
        if self.cur_task == 0: 
            self._network.init_unfreeze()
        else: 
            self._network.unfreeze_noise()
            
        # 2. Khởi tạo Optimizer và Scheduler
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

                # [FIX]: Dùng autocast an toàn để tránh lỗi Half vs Float
                with autocast('cuda', enabled=True):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            # Chạy backbone + analytic fc cũ
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        
                        # Chạy normal_fc mới (đang train)
                        logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        
                        # Fix lỗi kích thước và kiểu dữ liệu khi cộng logits
                        if logits2.shape[1] > logits1.shape[1]:
                            padding = torch.zeros((logits1.shape[0], logits2.shape[1] - logits1.shape[1]), 
                                                device=self.device, dtype=logits1.dtype)
                            logits1 = torch.cat([logits1, padding], dim=1)
                        
                        # Ép kiểu l1 về l2 để cộng an toàn
                        logits_final = logits2 + logits1.to(logits2.dtype)
                    else:
                        # Task 0: Chỉ dùng normal_fc
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    
                    loss = F.cross_entropy(logits_final, targets.long())

                # 3. Lan truyền ngược với Scaler (Mixed Precision)
                self.scaler.scale(loss).backward()
                
                # [QUAN TRỌNG]: Quy trình GPM cho GradScaler
                if self.cur_task > 0 and epoch >= WARMUP_EPOCHS:
                    # Phải unscale trước khi can thiệp vào Gradient (apply GPM)
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=current_scale)

                # 4. Cập nhật trọng số
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Thống kê
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            
            # Giải phóng cache định kỳ trong vòng lặp epoch
            if (epoch + 1) % 10 == 0: 
                self._clear_gpu()

        self.logger.info(info)
        
        # 5. [DỌN RÁC TRIỆT ĐỂ]: Trả lại VRAM cho bước nén (Compress)
        del optimizer
        del scheduler
        self._network.zero_grad(set_to_none=True)
        self._clear_gpu()
    @staticmethod
    def cat2order(targets, datamanger):
        return [datamanger.map_cat2order(t) for t in targets]

    def eval_task(self, test_loader):
        self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                logits = self._network(inputs)["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend(predicts.cpu().numpy())
                label.extend(targets.numpy())
        return calculate_class_metrics(pred, label)

    def after_train(self, data_manger):
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=0)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(eval_res['all_accy']*100, 2))
        self.logger.info(f'total acc: {self.total_acc}')
        self._clear_gpu()

    def save_check_point(self, path):
        torch.save(self._network.state_dict(), path)