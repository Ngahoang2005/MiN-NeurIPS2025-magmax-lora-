import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc
import os

from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler

from torch.amp import autocast, GradScaler 

# --- Main Class ---
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
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        self.scaler = GradScaler('cuda')
        
        # Buffer lưu trữ Mean của từng Class {class_id: mean_vector}
        self.class_means = {} 

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True
        
        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        # Train Noise Generator
        self.run(train_loader)
        self._network.after_task_magmax_merge()
        self._clear_gpu()
        
        # Fit RLS lần 1 (cho Task hiện tại)
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(train_loader_buf, test_loader)

        # Tính Simple Mean (Prototype)
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        clean_loader = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        self.compute_class_means(self._network, clean_loader)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        # Re-fit (Task 0 chưa có đồ cũ, nhưng vẫn chạy để đồng bộ)
        self.re_fit(clean_loader, test_loader)
        
        del train_set, test_set
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
        
        # 1. Fit FC (Mở rộng RLS cho Task mới)
        self.fit_fc(train_loader, test_loader)
        self._clear_gpu()

        # 2. Train Noise
        self._network.update_fc(self.increment)
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise() 
        self.run(train_loader_run)
        self._network.after_task_magmax_merge()
        self._clear_gpu()
        
        # Tính Mean cho Task MỚI
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        clean_loader = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        
        self.compute_class_means(self._network, clean_loader)

        # 3. Re-fit (Trộn Mean cũ + Data mới)
        self.re_fit(clean_loader, test_loader)
        
        del train_set, test_set
        self._clear_gpu()

    # =========================================================================
    #  HÀM TÍNH PROTOTYPE BẰNG SIMPLE MEAN
    # =========================================================================
    def compute_class_means(self, model, train_loader):
        """Tính trung bình feature cho mỗi class và lưu vào self.class_means"""
        model.eval()
        device = self.device
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                with autocast('cuda'):
                    feats = model.extract_feature(inputs).detach()
                # Normalize feature giúp Mean đại diện tốt hơn trên mặt cầu
                feats = F.normalize(feats, p=2, dim=1)
                all_features.append(feats)
                all_labels.append(targets)
        
        all_features = torch.cat(all_features).float()
        all_labels = torch.cat(all_labels).to(device)
        
        unique_classes = torch.unique(all_labels).cpu().numpy()
        
        # Tính Mean từng Class
        for cls in unique_classes:
            cls_mask = (all_labels == cls)
            cls_feats = all_features[cls_mask].to(device)
            
            # Tính Mean Vector
            mean_vector = torch.mean(cls_feats, dim=0) # [Feature_Dim]
            mean_vector = F.normalize(mean_vector, p=2, dim=0) # Normalize lại Mean
            
            # Lưu vào dictionary (CPU để tiết kiệm VRAM)
            self.class_means[cls] = mean_vector.cpu()
            
        self.logger.info(f"Computed Means for {len(unique_classes)} classes.")
        self._clear_gpu()

    # =========================================================================
    #  FIT_FC: CÔNG THỨC CHUẨN
    # =========================================================================
    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        if hasattr(self._network, 'set_grad_checkpointing'):
            self._network.set_grad_checkpointing(True)

        prog_bar = tqdm(range(self.fit_epoch))
        
        if self.cur_task == 0:
            current_total_classes = self.init_class
        else:
            current_total_classes = self.init_class + self.cur_task * self.increment

        with torch.no_grad():
            for _, epoch in enumerate(prog_bar):
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    targets_oh = torch.nn.functional.one_hot(targets, num_classes=current_total_classes).float()
                    self._network.fit(inputs, targets_oh)
                
                info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
                self.logger.info(info)
                prog_bar.set_description(info)
                if epoch % 5 == 0: gc.collect()

    # =========================================================================
    #  RE-FIT: DÙNG GAUSSIAN SAMPLING
    # =========================================================================
    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        if self.cur_task == 0:
            current_total_classes = self.init_class
        else:
            current_total_classes = self.init_class + self.cur_task * self.increment
            
        # 1. Feature Mới
        X_new_list, Y_new_list = [], []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                with autocast('cuda'):
                    features = self._network.extract_feature(inputs).detach().float()
                # Normalize Feature Mới cho đồng bộ với Mean đã lưu
                features = F.normalize(features, p=2, dim=1)
                X_new_list.append(features)
                Y_new_list.append(targets)
        
        X_new = torch.cat(X_new_list, dim=0)
        Y_new = torch.cat(Y_new_list, dim=0)
        
        # 2. Tạo Feature Cũ từ Mean (Gaussian Sampling)
        X_old_list, Y_old_list = [], []
        
        # Chỉ lấy các class thuộc Task Cũ (class id < start của task này)
        base_class_idx = current_total_classes - (self.increment if self.cur_task > 0 else 0)
        old_classes = [c for c in self.class_means.keys() if c < base_class_idx]
        
        if len(old_classes) > 0:
            # Cân bằng: Tổng số mẫu cũ ~ 50% Tổng số mẫu mới
            total_new_samples = X_new.size(0)
            samples_per_old_class = int((total_new_samples * 0.5) / len(old_classes)) 
            samples_per_old_class = max(20, samples_per_old_class) # Đảm bảo ít nhất 20 mẫu/lớp
            
            sigma = 0.05 # Độ lệch chuẩn nhỏ để tạo đám mây xung quanh Mean
            
            for cls in old_classes:
                mean_vec = self.class_means[cls].to(self.device) # [Dim]
                
                # Gaussian Sampling: N(Mean, Sigma)
                generated_feats = torch.normal(mean=mean_vec.repeat(samples_per_old_class, 1), std=sigma)
                generated_feats = F.normalize(generated_feats, p=2, dim=1)
                
                label_vec = torch.full((samples_per_old_class,), cls, dtype=torch.long, device=self.device)
                
                X_old_list.append(generated_feats)
                Y_old_list.append(label_vec)
                
            X_old = torch.cat(X_old_list, dim=0)
            Y_old = torch.cat(Y_old_list, dim=0)
            
            X_total = torch.cat([X_new, X_old], dim=0)
            Y_total = torch.cat([Y_new, Y_old], dim=0)
            
            self.logger.info(f"Re-fit Data: New {X_new.size(0)} + Old {X_old.size(0)} (Gaussian Generated)")
        else:
            X_total, Y_total = X_new, Y_new

        # 3. Fit
        Y_total_oh = torch.nn.functional.one_hot(Y_total, num_classes=current_total_classes).float()
        
        batch_size_fit = 4096
        total_samples = X_total.size(0)
        perm = torch.randperm(total_samples)
        
        info = f"Task {self.cur_task} --> Re-fit RLS with Gaussian Prototypes..."
        self.logger.info(info)
        
        with torch.no_grad():
            for i in tqdm(range(0, total_samples, batch_size_fit), desc="Re-fitting"):
                idx = perm[i : i + batch_size_fit]
                x_batch = X_total[idx]
                y_batch = Y_total_oh[idx]
                self._network.fit(x_batch, y_batch)
        self._clear_gpu()

    # =========================================================================
    #  [FIXED] RUN: RESTORE USER'S ORIGINAL EPOCH/LR LOGIC
    # =========================================================================
    def run(self, train_loader):
        # Logic gốc của bạn
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = 5
            lr = self.lr * 0.1
            weight_decay = self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
            
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        self._clear_gpu()
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        if hasattr(self._network, 'set_grad_checkpointing'):
            self._network.set_grad_checkpointing(True)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)

        for _, epoch in enumerate(prog_bar):
            losses, correct, total = 0.0, 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs2['logits'] + outputs1['logits']
                        loss = F.cross_entropy(logits_final, targets.long())
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs["logits"]
                        loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, logits_final, loss 
            
            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} Epoch {}/{} => Loss {:.3f}, Acc {:.2f}".format(self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
            if epoch % 5 == 0: gc.collect()
        self._clear_gpu()

    def after_train(self, data_manger):
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        print('total acc: {}'.format(self.total_acc))
        del test_set

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
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
            "task_confusion": task_info['task_confusion_matrices']
        }