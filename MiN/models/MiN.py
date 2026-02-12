import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc  # [ADDED] Để dọn rác bộ nhớ
import os
import matplotlib.pyplot as plt
from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# [ADDED] Import Mixed Precision
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
        
        # [ADDED] Scaler cho Mixed Precision
        self.scaler = GradScaler('cuda')

    def _clear_gpu(self):
        # [ADDED] Hàm dọn dẹp GPU
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
        if self.cur_task >= 0:
            self.analyze_entropy_accuracy(test_loader)
            self.analyze_universal_correlation(test_loader)
            self.visualize_expert_orthogonality()
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                if self.cur_task > 0:
                    outputs = model.forward_tuna_combined(inputs)
                else:
                    model.set_noise_mode(-2)
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
    def merge_noise_experts(self):
        print(f"\n>>> Merging Noise Experts (TUNA EMR) for Task {self.cur_task}...")
        
        # 1. Truy cập vào Backbone
        # self._network là MiNbaseNet -> .backbone là ViT
        if hasattr(self._network.backbone, 'noise_maker'):
            
            # 2. Duyệt qua từng lớp PiNoise trong Backbone
            for m in self._network.backbone.noise_maker:
                
                # 3. Gọi hàm merge_noise() (Hàm này nằm trong PiNoise như bạn nói)
                m.merge_noise()
                
        self._clear_gpu()

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, _, _ = data_manger.get_task_list(0)
        self.logger.info(f"Task 0 Order: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=train_list) # Test on current task
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        
        self.run(train_loader) # Train SGD
        self.merge_noise_experts() # Merge Task 0
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        
        # Fit RLS
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader_buf = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self._network.set_noise_mode(-2)
        self.fit_fc(train_loader_buf, test_loader_buf)

        # Refit (Optional)
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        self.re_fit(train_loader_clean, test_loader_buf)
        
        del train_set, test_set, train_set_clean; self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"Task {self.cur_task} Order: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        
        # [SỬA LỖI QUAN TRỌNG] Phải update FC (mở rộng mạng) TRƯỚC KHI gọi fit_fc
        # Nếu không, one_hot sẽ bị lỗi index out of bounds vì nhãn mới > số class cũ
        self._network.update_fc(self.increment)
        self._network.update_noise()

        # 1. Fit RLS Universal (Giờ mạng đã đủ lớn để chứa class mới)
        self._network.set_noise_mode(-2)
        self.fit_fc(train_loader, test_loader)
        
        # 2. Train SGD Specific
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader_run)
        self._network.extend_task_prototype(prototype)
        
        self.run(train_loader_run) # Train SGD
        self.merge_noise_experts()
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader_run)
        self._network.update_task_prototype(prototype)

        # 3. Final Refit
        del train_set
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
            
        self._network.set_noise_mode(self.cur_task)
        self.re_fit(train_loader_clean, test_loader)

        del train_set_clean, test_set
        self._clear_gpu()
    
    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        if self.cur_task > 0:
            decay = 0.9  # Giữ lại 90% kiến thức cũ
            self._network.H *= decay
            self._network.Hy *= decay
        else:
            self._network.H.zero_()
            self._network.Hy.zero_()
        print(">>> [Fast RLS] Accumulating Statistics form Train Loader...")
        
        # Nếu Dataloader có Data Augmentation, bạn có thể chạy nhiều epoch để lấy trung bình thống kê tốt hơn.
        # Nếu không Augmentation, chỉ cần chạy 1 epoch là đủ (Toán học chứng minh là tương đương).
        # Ở đây tôi giữ vòng lặp epoch nhưng thường 1 epoch là đủ cho RLS dạng này.
        
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Chuyển target sang one-hot global
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            
            # [BƯỚC 1] Chỉ tích lũy (Accumulate), cực nhanh
            self._network.fit_batch(inputs, targets)
            
        # [BƯỚC 2] Giải hệ phương trình 1 lần duy nhất sau khi đã xem hết dữ liệu
        print(f">>> [RLS Update] Task {self.cur_task} - Accumulating...")
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit_batch(inputs, targets)
            
        # Giải hệ phương trình với Lambda nhỏ hơn để tăng biên độ Logit
        # (Sửa lambda_reg xuống 1.0 trong inc_net.py)
        self._network.update_analytical_weights()
        self._clear_gpu()

    # --------------------------------------------------------------------------
    # 2. RE-FIT FUNCTION (Dùng cho tập Clean/Buffer cuối task)
    # --------------------------------------------------------------------------
    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # Reset lại để fit riêng cho tập Clean này (Refine weights)
        self._network.H.zero_()
        self._network.Hy.zero_()

        print(">>> [Fast RLS] Re-fitting on Clean Data...")
        prog_bar = tqdm(train_loader)
        
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            
            # [BƯỚC 1] Tích lũy
            self._network.fit_batch(inputs, targets)

        # [BƯỚC 2] Giải hệ phương trình
        self._network.update_analytical_weights()

        self.logger.info(f"Task {self.cur_task} --> Classifier Refined on Clean Data!")
        self._clear_gpu()
    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
            
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise() # Expert đã được Hot-Init
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        
        # Mode = Current Task (để train Expert này)
        self._network.set_noise_mode(self.cur_task) 
        
        for _, epoch in enumerate(prog_bar):
            losses = 0.0; correct = 0; total = 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 

                with autocast('cuda'):
                    # 1. Main Loss (Ensemble logic)
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits_final = logits2 + logits1
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits_final = outputs["logits"]
                    
                    loss_ce = F.cross_entropy(logits_final, targets.long())

                    # 2. ORTHOGONAL LOSS (Ép vuông góc)
                    loss_orth = torch.tensor(0.0).to(self.device)
                    if self.cur_task > 0:
                        # Duyệt qua các tầng noise
                        for m in self._network.backbone.noise_maker:
                            # Lấy vector Expert hiện tại (Flatten)
                            curr_mu = m.mu[self.cur_task].weight.flatten()
                            
                            # Lấy các vector Expert cũ
                            prev_mus = []
                            for t in range(self.cur_task):
                                prev_mus.append(m.mu[t].weight.flatten())
                            
                            if len(prev_mus) > 0:
                                prev_mus_stack = torch.stack(prev_mus) # [N_prev, Dim]
                                
                                # Tính Cosine Similarity
                                # Normalize để tránh scale ảnh hưởng
                                curr_norm = F.normalize(curr_mu.unsqueeze(0), p=2, dim=1)
                                prev_norm = F.normalize(prev_mus_stack, p=2, dim=1)
                                
                                # Matrix Mul: [1, Dim] @ [Dim, N_prev] -> [1, N_prev]
                                cos_sim = torch.mm(curr_norm, prev_norm.t())
                                
                                # Loss = Tổng trị tuyệt đối của Cosine (càng gần 0 càng tốt)
                                loss_orth += torch.sum(torch.abs(cos_sim))

                    # Tổng hợp Loss
                    lambda_orth = 0.5 # Hệ số phạt (có thể chỉnh 0.1 -> 1.0)
                    loss = loss_ce + lambda_orth * loss_orth

                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = f"Task {self.cur_task} SGD: Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, Acc {train_acc:.2f}%"
            self.logger.info(info); prog_bar.set_description(info)
            if epoch % 5 == 0: self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                if self.cur_task > 0:
                    outputs = model.forward_tuna_combined(inputs)
                else:
                    self._network.set_noise_mode(-2)
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
        
        # 1. Thu thập features (CHUYỂN VỀ CPU NGAY LẬP TỨC)
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                
                # Dùng autocast khi extract feature để nhanh hơn
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                
                # .detach().cpu() là chìa khóa để tránh OOM
                features.append(feature.detach().cpu())
        
        # 2. Concat trên CPU (RAM thường lớn hơn VRAM)
        all_features = torch.cat(features, dim=0)
        
        # 3. Tính Mean (Vẫn tính trên CPU hoặc đưa về GPU nếu cần)
        # Vì chỉ tính mean của 1 tensor lớn, đưa về GPU tính sẽ nhanh, 
        # nhưng nếu tensor quá lớn > VRAM thì tính trên CPU luôn.
        # Ở đây tôi để tính trên GPU cho nhanh, nếu vẫn OOM thì xóa .to(self.device)
        prototype = torch.mean(all_features, dim=0).to(self.device)
        
        self._clear_gpu()
        return prototype
    # =========================================================================
    # DEBUG TOOLS: ENTROPY, ACCURACY & ORTHOGONALITY VISUALIZATION
    # =========================================================================

    def analyze_entropy_accuracy(self, test_loader):
        """Vẽ biểu đồ tương quan giữa Độ tự tin (Entropy) và Độ chính xác (Accuracy)"""
        self._network.eval()
        all_entropies = []
        all_correct_flags = []
    
        print(f">>> [DEBUG] Analyzing Entropy vs Accuracy for Task {self.cur_task}...")
        max_samples = 2000 # Lấy mẫu vừa đủ để vẽ nhanh
        current_samples = 0
    
        with torch.no_grad():
            for _, inputs, targets in tqdm(test_loader, desc="Collecting Entropy Data"):
                if current_samples >= max_samples: break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 1. Kích hoạt Expert hiện tại để kiểm tra
                self._network.set_noise_mode(self.cur_task)
                
                # 2. Forward thủ công qua Shared W
                feat = self._network.extract_feature(inputs)
                feat = self._network.buffer(feat)
                logits = self._network.forward_fc(feat)
                
                # 3. Tính Entropy
                prob = torch.softmax(logits, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                
                # 4. Check đúng/sai
                predicts = torch.max(logits, dim=1)[1]
                correct = (predicts == targets).float()
    
                all_entropies.extend(entropy.cpu().numpy())
                all_correct_flags.extend(correct.cpu().numpy())
                current_samples += inputs.shape[0]
    
        self._plot_bin_graph(all_entropies, all_correct_flags, mode="Specific")

    def analyze_universal_correlation(self, test_loader):
        """Kiểm tra nhánh Universal hoạt động thế nào"""
        self._network.eval()
        all_entropies = []
        all_correct_flags = []
        print(f">>> [DEBUG] Analyzing Universal Branch...")
        
        max_samples = 2000
        current_samples = 0
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(test_loader, desc="Collecting Universal Data"):
                if current_samples >= max_samples: break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Set Universal Mode
                self._network.set_noise_mode(-2)
                
                feat = self._network.extract_feature(inputs)
                feat = self._network.buffer(feat)
                logits = self._network.forward_fc(feat)
                
                prob = torch.softmax(logits, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                predicts = torch.max(logits, dim=1)[1]
                correct = (predicts == targets).float()
    
                all_entropies.extend(entropy.cpu().numpy())
                all_correct_flags.extend(correct.cpu().numpy())
                current_samples += inputs.shape[0]
    
        self._plot_bin_graph(all_entropies, all_correct_flags, mode="Universal")

    def _plot_bin_graph(self, entropies, correct_flags, mode="Specific", num_bins=10):
        try:
            entropies = np.array(entropies)
            correct_flags = np.array(correct_flags)
            if len(entropies) == 0: return

            # Chia bins từ min entropy đến max entropy
            min_e, max_e = entropies.min(), entropies.max()
            if min_e == max_e: max_e += 1e-5
            
            bin_edges = np.linspace(min_e, max_e, num_bins + 1)
            accuracies = []
            avg_entropies = []
        
            for i in range(num_bins):
                mask = (entropies >= bin_edges[i]) & (entropies < bin_edges[i+1])
                if mask.any():
                    accuracies.append(correct_flags[mask].mean() * 100) # % Accuracy
                    avg_entropies.append((bin_edges[i] + bin_edges[i+1]) / 2)
        
            # Vẽ biểu đồ
            plt.figure(figsize=(8, 6))
            color = 'red' if mode == "Specific" else 'blue'
            plt.plot(accuracies, avg_entropies, marker='o', color=color, linewidth=2, label=f'{mode} Branch')
            
            plt.xlabel('Accuracy (%)', fontsize=12)
            plt.ylabel('Entropy (Uncertainty)', fontsize=12)
            plt.title(f'Debug: {mode} Confidence (Task {self.cur_task})', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.gca().invert_yaxis() # Đảo trục Y: Entropy thấp (Tốt) ở trên cao
            plt.legend()
            
            filename = f'debug_{mode.lower()}_task_{self.cur_task}.png'
            plt.savefig(filename)
            plt.close()
            print(f">>> [PLOT] Saved chart to: {filename}")
        except Exception as e:
            print(f">>> [PLOT ERROR] {e}")

    def visualize_expert_orthogonality(self):
        """Vẽ Heatmap thể hiện độ tương quan giữa các Expert"""
        if self.cur_task == 0: return # Chưa có gì để so sánh
        
        print(f">>> [DEBUG] Visualizing Expert Orthogonality...")
        try:
            # Thu thập trọng số Mu từ tất cả Expert
            mus = []
            with torch.no_grad():
                for i in range(self.cur_task + 1):
                    # Lấy trung bình trọng số của các layer noise để đại diện cho task đó
                    # Hoặc lấy layer cuối cùng (thường chứa thông tin ngữ nghĩa cao nhất)
                    layer_idx = self._network.backbone.layer_num - 1
                    expert_mu = self._network.backbone.noise_maker[layer_idx].mu[i].weight.data.flatten()
                    mus.append(expert_mu)
            
            if not mus: return

            # Stack lại: [Num_Tasks, Dim]
            mus_tensor = torch.stack(mus)
            
            # Normalize vector về đơn vị
            mus_norm = F.normalize(mus_tensor, p=2, dim=1)
            
            # Tính Ma trận Cosine Similarity: [Num_Tasks, Num_Tasks]
            # Heatmap[i, j] = Cosine(Expert_i, Expert_j)
            similarity_matrix = torch.mm(mus_norm, mus_norm.t()).cpu().numpy()
            
            # Vẽ Heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Cosine Similarity')
            plt.title(f'Expert Orthogonality Matrix (Task 0-{self.cur_task})')
            plt.xlabel('Expert ID')
            plt.ylabel('Expert ID')
            
            # Hiển thị số trên ô
            for i in range(len(similarity_matrix)):
                for j in range(len(similarity_matrix)):
                    text = f"{similarity_matrix[i, j]:.2f}"
                    color = "white" if similarity_matrix[i, j] < 0.5 else "black"
                    plt.text(j, i, text, ha="center", va="center", color=color)
            
            filename = f'debug_orthogonality_task_{self.cur_task}.png'
            plt.savefig(filename)
            plt.close()
            print(f">>> [PLOT] Saved orthogonality map to: {filename}")
            
        except Exception as e:
            print(f">>> [PLOT ERROR] Could not visualize orthogonality: {e}")