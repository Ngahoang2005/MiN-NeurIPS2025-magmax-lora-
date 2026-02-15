import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc 
import os

from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics
import torch
import matplotlib.pyplot as plt
import numpy as np
# Import Mixed Precision
from torch.amp import autocast, GradScaler

EPSILON = 1e-8

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
        self.class_acc = []
        self.task_acc = []
        
        # Scaler cho Mixed Precision
        self.scaler = GradScaler('cuda')

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        
        del test_set

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

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        self._clear_gpu()
        
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.9)
        
        
        self._clear_gpu()
        
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        kurtosis_score = self.diagnose_noise(self._network, train_loader, self.device)
        print(f"Kurtosis Score for Task {self.cur_task}: {kurtosis_score:.2f}")
        self.fit_fc(train_loader, test_loader)

        train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader, test_loader)
        #self.check_rls_quality()
        del train_set, test_set
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.test_loader = test_loader

        # [FIX 1: QUAN TRỌNG] Phải update FC (mở rộng class) TRƯỚC KHI fit
        # Để fit_fc biết được đúng số lượng class mới
        self._network.update_fc(self.increment)
        
        # Update Noise Generator cho task mới
        self._network.update_noise()

        # [STEP 1] Analytic Learning (RLS)
        # Fit trên dữ liệu task mới (đồng thời tích lũy vào bộ nhớ A_global, B_global)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False
        kurtosis_score = self.diagnose_noise(self._network, train_loader, self.device)
        print(f"Kurtosis Score for Task {self.cur_task}: {kurtosis_score:.2f}") 
        self.fit_fc(train_loader, test_loader)

        # [STEP 2] Training Noise (SGD)
        # Tạo lại loader với batch_size nhỏ hơn cho việc train noise
        train_loader_sgd = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.num_workers)
        
        self._clear_gpu()
        self.run(train_loader_sgd)
        
        # Thu thập GPM Projection sau khi train xong noise
        self._network.collect_projections(mode='threshold', val=0.9)
        self._clear_gpu()

        del train_set

        # [STEP 3] Re-Fit Analytic Classifier (Final Polish)
        # Dùng tập train không augmentation để chốt hạ classifier
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader_no_aug, test_loader)
        
        del train_set_no_aug, test_set
        self._clear_gpu()

    # def fit_fc(self, train_loader, test_loader):
    #     # [FIX 2: MEMORY ACCUMULATION]
    #     # RLS cần nhớ ma trận tương quan (A) và (B) của các task cũ.
    #     # Nếu tính lại từ đầu, model sẽ quên sạch quá khứ.
        
    #     self._network.eval()
    #     self._network.to(self.device)
        
    #     # 1. Xác định kích thước feature
    #     with torch.no_grad():
    #         dummy_input = next(iter(train_loader))[1].to(self.device)
    #         dummy_feat = self._network.extract_feature(dummy_input)
    #         if hasattr(self._network, 'buffer'):
    #             dummy_feat = self._network.buffer(dummy_feat.float())
    #         feat_dim = dummy_feat.shape[1]
        
    #     # Lấy tổng số class ĐÃ ĐƯỢC MỞ RỘNG
    #     num_classes = self._network.known_class
        
    #     # 2. Khởi tạo Global Memory nếu chưa có (lưu trong self của MinNet để persist qua các task)
    #     if not hasattr(self, 'A_global'):
    #         print("--> Initializing Global RLS Memory...")
    #         self.A_global = torch.zeros((feat_dim, feat_dim), device=self.device, dtype=torch.float32)
    #         self.B_global = torch.zeros((feat_dim, 0), device=self.device, dtype=torch.float32)

    #     # Mở rộng B_global nếu số class tăng lên
    #     current_B_cols = self.B_global.shape[1]
    #     if num_classes > current_B_cols:
    #         diff = num_classes - current_B_cols
    #         expansion = torch.zeros((feat_dim, diff), device=self.device, dtype=torch.float32)
    #         self.B_global = torch.cat([self.B_global, expansion], dim=1)
            
    #     print(f"--> Accumulating Statistics for Task {self.cur_task} (Total Classes: {num_classes})...")
        
    #     # 3. Tích lũy thống kê (Chỉ cộng thêm phần của Task mới)
    #     # Lưu ý: A_new và B_new là thống kê của RIÊNG dữ liệu hiện tại
    #     A_new = torch.zeros((feat_dim, feat_dim), device=self.device, dtype=torch.float32)
    #     B_new = torch.zeros((feat_dim, num_classes), device=self.device, dtype=torch.float32)
    #     fit_epochs = self.fit_epoch 
        
    #     for epoch in range(fit_epochs):
    #         with torch.no_grad():
    #             for i, (_, inputs, targets) in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1}")):
    #                 inputs, targets = inputs.to(self.device), targets.to(self.device)
    #                 if targets.dim() > 1: targets = targets.view(-1)
                    
    #                 # Forward
    #                 features = self._network.extract_feature(inputs).float()
    #                 features = self._network.buffer(features)
                    
    #                 # One-hot (đã an toàn vì num_classes được update từ bước update_fc)
    #                 targets_oh = F.one_hot(targets.long(), num_classes=num_classes).float()
                    
    #                 A_new += features.T @ features
    #                 B_new += features.T @ targets_oh
        
    #     # 4. Cộng vào bộ nhớ toàn cục
    #     self.A_global += A_new
    #     self.B_global += B_new
        
    #     # 5. Giải hệ phương trình trên bộ nhớ toàn cục
    #     # W = (A_global + gamma * I)^-1 @ B_global
    #     print("--> Solving Global Linear System...")
    #     gamma = self.args['gamma']
    #     I = torch.eye(feat_dim, device=self.device, dtype=torch.float32)
    #     A_reg = self.A_global + gamma * I
        
    #     try:
    #         W = torch.linalg.solve(A_reg, self.B_global)
    #     except RuntimeError:
    #         W = torch.linalg.pinv(A_reg) @ self.B_global
            
    #     # 6. Gán trọng số
    #     if self._network.weight.shape != W.shape:
    #         self._network.weight = torch.zeros_like(W)
    #     self._network.weight.data = W
            
    #     print("--> Analytic Learning Finished.")
    #     self._clear_gpu()

    #fit mmcm
    def fit_fc(self, train_loader, test_loader, update_archive=True):
        self._network.eval()
        self._network.to(self.device)

        # --- HYPERPARAMS (Theo yêu cầu của bạn) ---
        SIGMA = 5.0   # Bắt đầu với 5.0, giảm dần nếu cần
        OMEGA = 0.4   # Mixture ratio
        LAMBDA = 100.0 # Hệ số khởi tạo P (Tương đương 1/lambda nhỏ trong Ridge)
        
        # 1. Init Feature Dim
        with torch.no_grad():
            dummy_input = next(iter(train_loader))[1].to(self.device)
            dummy_feat = self._network.extract_feature(dummy_input).double()
            if hasattr(self._network, 'buffer'):
                dummy_feat = self._network.buffer(dummy_feat.float()).double()
            feat_dim = dummy_feat.shape[1]

        num_classes = self._network.known_class

        # 2. Khởi tạo Global Covariance Matrix (P / Omega)
        if not hasattr(self, 'P_global'):
            print(f"--> [Recursive] Init P_global (Scale={LAMBDA})...")
            # Khởi tạo P = Lambda * I (Lambda lớn -> Quên nhanh lúc đầu)
            self.P_global = torch.eye(feat_dim, device=self.device, dtype=torch.float64) * LAMBDA
        
        # Reset Weight nếu cần (hoặc giữ nguyên để học tiếp)
        if self._network.weight.shape[1] == 0:
             self._network.weight = torch.zeros((feat_dim, num_classes), device=self.device, dtype=torch.float64)
        
        # Expand Weight
        if num_classes > self._network.weight.shape[1]:
            diff = num_classes - self._network.weight.shape[1]
            tail = torch.zeros((feat_dim, diff), device=self.device, dtype=torch.float64)
            self._network.weight = torch.cat([self._network.weight, tail], dim=1)
            
        # Đảm bảo weight là double
        if self._network.weight.dtype != torch.float64:
            self._network.weight = self._network.weight.double()

        # 3. Training Loop (Recursive)
        # Recursive chỉ cần chạy 1 Epoch là hội tụ (One-pass learning)
        # Nếu update_archive=False (Refit), cũng chạy 1 epoch.
        
        print(f"--> [Recursive MMCC] Processing Stream (Batch Size per step: {train_loader.batch_size})...")
        
        # [QUAN TRỌNG] Tắt Autocast để chạy Double Precision (float64)
        with autocast('cuda', enabled=False):
            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(tqdm(train_loader, desc="Streaming")):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Extract Feature
                    features = self._network.extract_feature(inputs).double()
                    if hasattr(self._network, 'buffer'):
                        features = self._network.buffer(features.float()).double()
                    
                    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).double()
                    
                    # Gọi hàm Recursive (Update P_global in-place và trả về)
                    # Hàm này sẽ loop bên trong batch để đảm bảo đúng quy trình
                    self.P_global = self._network.fit_recursive_mmcc_batch(
                        features, targets_oh, self.P_global, sigma=SIGMA, omega=OMEGA
                    )

        # Cast về float32 sau khi xong để tương thích
        self._network.weight = self._network.weight.float()
        
        # [LƯU Ý] Recursive RLS tự động cập nhật P_global liên tục
        # Không cần bước "Giải hệ phương trình" cuối cùng như Batch RLS.
        # W đã được cập nhật liên tục trong vòng lặp rồi.
        
        print("--> Recursive Update Finished.")
        self._clear_gpu()
    def re_fit(self, train_loader, test_loader):
        # re_fit dùng chung logic với fit_fc
        print(f"--> Refitting Task {self.cur_task} (No Augmentation)...")
        self.fit_fc(train_loader, test_loader)
       
    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

        current_scale = 0.85
        
        # Freeze/Unfreeze Logic
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
        max_beta = 1e-4 # [LƯU Ý] Chỉnh lại max_beta tùy ý bạn (1e-4 hoặc 1e-5)
        
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            ce_losses = 0.0 # Theo dõi riêng CE
            kl_losses = 0.0 # Theo dõi riêng KL
            correct, total = 0, 0

            beta_current = max_beta * min(1.0, epoch / (epochs / 2 + 1e-6))

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad(set_to_none=True) 

                # 1. FORWARD
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        logits2, batch_kl = self._network.forward_with_ib(inputs)
                        logits_final = logits2 + logits1
                    else:
                        logits_final, batch_kl = self._network.forward_with_ib(inputs)

                # 2. CALC LOSS
                logits_final = logits_final.float() 
                if targets.dim() > 1: targets = targets.reshape(-1)
                targets = targets.long()

                ce_loss = F.cross_entropy(logits_final, targets)
                loss = ce_loss + beta_current * batch_kl

                # 3. BACKWARD
                self.scaler.scale(loss).backward()
                
                if self.cur_task > 0 and epoch >= WARMUP_EPOCHS:
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=current_scale)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # 4. METRICS & LOGGING
                losses += loss.item()
                ce_losses += ce_loss.item()      # Cộng dồn CE
                kl_losses += batch_kl.item()     # Cộng dồn KL
                
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                del inputs, targets, loss, logits_final, batch_kl

                # In bảng Noise mỗi 50 batch
                if i % 50 == 0:
                     if self.cur_task > 0 or (self.cur_task == 0 and epoch == epochs - 1):
                        self.print_noise_status()

            scheduler.step()
            train_acc = 100. * correct / total

            # [HIỂN THỊ CHI TIẾT LOSS]
            # L: Tổng | CE: CrossEntropy (Dự đoán) | KL: IB Loss (Nén)
            info = "T {} | Ep {} | L {:.3f} (CE {:.3f} | KL {:.1f}) | Acc {:.2f}".format(
                self.cur_task, epoch + 1, 
                losses / len(train_loader), 
                ce_losses / len(train_loader),
                kl_losses / len(train_loader),
                train_acc
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            
            if epoch % 5 == 0:
                self._clear_gpu()
    
    
    def print_noise_status(self):
        print("\n" + "="*85)
        print(f"{'Layer':<10} | {'Signal':<10} | {'Noise':<10} | {'SNR':<10} | {'Sigma':<10} | {'Scale':<10} | {'Status'}")
        print("-" * 85)
        
        # Lấy danh sách các lớp Noise từ backbone
        # Lưu ý: Cấu trúc backbone của bạn có thể khác, hãy đảm bảo path đúng
        # Ví dụ: self._network.backbone.noise_maker
        noise_layers = []
        if hasattr(self._network.backbone, 'noise_maker'):
             noise_layers = self._network.backbone.noise_maker
        
        for i, layer in enumerate(noise_layers):
            # Kiểm tra xem lớp đó có biến last_debug_info không (đã thêm ở bước trước)
            if not hasattr(layer, 'last_debug_info') or not layer.last_debug_info: 
                continue
            
            info = layer.last_debug_info
            
            # Đánh giá trạng thái
            snr = info['snr']
            if snr < 1.0: status = "TOXIC ☠️"       # Nhiễu to hơn tín hiệu
            elif snr < 10.0: status = "HEAVY ⚠️"    # Nhiễu nặng
            elif snr > 1000.0: status = "USELESS 💤" # Nhiễu quá bé
            else: status = "GOOD ✅"                # 10 < SNR < 1000
            
            print(f"L{i:<9} | {info['signal']:.4f}     | {info['noise']:.4f}     | {snr:.1f}       | {info['sigma']:.4f}     | {info['scale']:.4f}     | {status}")
        print("="*85 + "\n")
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
    

    # Thêm 'self' vào đầu
    def diagnose_noise(self, network, data_loader, device='cuda'):
        """
        Phân tích độ nhiễu của dữ liệu dựa trên mô hình RLS sơ bộ.
        Trả về: Kurtosis score và lời khuyên.
        """
        print("\n" + "="*40)
        print(">>> DATA NOISE DIAGNOSIS START <<<")
        
        network.eval()
        network.to(device)
        
        all_features = []
        all_targets = []
        
        # 1. Thu thập toàn bộ dữ liệu (hoặc 1 phần nếu data quá lớn)
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Extract Feature
                feats = network.extract_feature(inputs).float()
                if hasattr(network, 'buffer'):
                    feats = network.buffer(feats)
                    
                targets_oh = torch.nn.functional.one_hot(
                    targets.long(), num_classes=network.known_class
                ).float()
                
                all_features.append(feats)
                all_targets.append(targets_oh)
                
                if i > 50: break # Chỉ cần check 50 batch đầu là đủ thống kê
                
        X = torch.cat(all_features, dim=0)
        Y = torch.cat(all_targets, dim=0)
        
        # 2. Fit nhanh một cái RLS cơ bản (MSE) để tìm mặt phẳng tham chiếu
        # W_temp = (X^T X + I)^-1 X^T Y
        # Ta dùng lstsq cho nhanh
        print("--> Fitting temporary RLS to check residuals...")
        try:
            # Giải hệ phương trình tuyến tính để tìm W tạm
            # Dùng driver 'gels' cho tổng quát
            W_temp = torch.linalg.lstsq(X, Y).solution
        except:
            # Fallback nếu lỗi
            I = torch.eye(X.shape[1], device=device)
            W_temp = torch.linalg.solve(X.T @ X + 1.0 * I, X.T @ Y)
            
        # 3. Tính sai số (Residuals)
        # E = Y - X * W
        preds = X @ W_temp
        errors = Y - preds
        
        # Lấy giá trị tuyệt đối hoặc bình phương lỗi
        # Ta quan tâm phân phối của nó
        error_vals = errors.flatten().cpu().numpy()
        
        # 4. Tính toán chỉ số Kurtosis (Độ nhọn)
        # Kurtosis cao = Có nhiều Outliers (Đuôi dài)
        mean_e = np.mean(error_vals)
        std_e = np.std(error_vals)
        
        # Công thức Kurtosis chuẩn hóa (Fisher definition: Normal = 0)
        # Hoặc Pearson definition: Normal = 3. Ở đây dùng Pearson cho dễ hình dung.
        numerator = np.mean((error_vals - mean_e)**4)
        denominator = std_e**4
        kurtosis = numerator / denominator
        
        # 5. Phân tích thêm Outlier Ratio
        # Outlier là những điểm có lỗi > 3 * STD (quy tắc 3-sigma)
        threshold = 3 * std_e
        n_outliers = np.sum(np.abs(error_vals - mean_e) > threshold)
        outlier_ratio = (n_outliers / len(error_vals)) * 100
        
        print("-" * 40)
        print(f"Dataset Stats:")
        print(f"  - Mean Error: {mean_e:.5f}")
        print(f"  - Std Error:  {std_e:.5f}")
        print(f"  - Max Error:  {np.max(np.abs(error_vals)):.5f}")
        print(f"  - Kurtosis:   {kurtosis:.2f} (Gaussian approx 3.0)")
        print(f"  - Outliers (>3std): {outlier_ratio:.2f}%")
        print("-" * 40)
        
        # 6. Đưa ra lời khuyên
        print(">>> CONCLUSION:")
        if kurtosis > 5.0 or outlier_ratio > 1.0:
            print("🔴 HIGH NOISE DETECTED (Non-Gaussian).")
            print("   -> Feature/Label bị nhiễu hoặc có nhiều Hard Samples cực đoan.")
            print("   -> KHUYÊN DÙNG: MMCC (Robust RLS).")
            print("   -> Config: Tăng Omega lên 0.6, Sigma = 3.0")
        elif kurtosis < 2.0:
            print("🟡 LOW NOISE (Platykurtic - Phân phối bẹt).")
            print("   -> Dữ liệu quá sạch hoặc phân tán đều.")
            print("   -> KHUYÊN DÙNG: RLS Thường (Acc cao hơn).")
        else: # 2.0 <= Kurtosis <= 5.0
            print("🟢 CLEAN DATA (Gaussian-like).")
            print("   -> Dữ liệu chuẩn, nhiễu ngẫu nhiên.")
            print("   -> KHUYÊN DÙNG: RLS Thường (Tốt nhất).")
            
        print("="*40 + "\n")
        
        return kurtosis