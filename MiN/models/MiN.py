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
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        # [FIX]: Set num_workers=0 để tránh lỗi DataLoader worker killed
        self.num_workers = 0 

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
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info(f"task_list: {train_list_name}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        # 1. RUN (Train Backbone)
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # 2. FIT (Train Classifier bằng RLS gốc - 3 Epoch)
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        self.fit_fc(train_loader_buf)

        # 3. RE-FIT (Lưu stats chuẩn bị cho DPCR Task sau)
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_no_aug)
        
        del train_set, test_set, train_set_no_aug
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"task_list: {train_list_name}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        # Lưu Old Model (Trên CPU để tiết kiệm VRAM)
        self._old_network = copy.deepcopy(self._network).eval().cpu()

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        # 1. FIT (Warm-up / Teacher Classifier bằng RLS gốc)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        
        self.fit_fc(train_loader) 
        self._network.update_fc(self.increment)

        # 2. RUN (Train Backbone)
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()
        
        print(f"--> Start Training Backbone Task {self.cur_task}...")
        self.run(train_loader_run)
        
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        del train_loader_run

        # 3. RE-FIT (DPCR Update: TSSP -> CIP -> CN)
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_no_aug)
        
        # Cleanup
        self._old_network = None
        del train_set, test_set, train_set_no_aug
        self._clear_gpu()

    # def calculate_drift(self, clean_loader):
    #     """Tính Drift P trên Backbone Space (768-d) + CHECK DRIFT"""
    #     self._network.eval()
        
    #     # Đưa old_network lên GPU
    #     self._old_network.to(self.device)
    #     self._old_network.eval()
        
    #     dim = self._network.feature_dim
    #     # Dùng double precision để tính toán chính xác nhất
    #     XtX = torch.zeros(dim, dim, device=self.device, dtype=torch.float64)
    #     XtY = torch.zeros(dim, dim, device=self.device, dtype=torch.float64)
    #     ref_dtype = next(self._network.backbone.parameters()).dtype
        
    #     # Biến để monitor độ trôi
    #     diff_sum = 0.0
    #     count = 0
        
    #     with torch.no_grad():
    #         for _, inputs, _ in clean_loader:
    #             inputs = inputs.to(self.device, dtype=ref_dtype)
                
    #             # [QUAN TRỌNG]: Đảm bảo backbone trả về feature ĐÃ CÓ NOISE
    #             # Nếu backbone code của bạn cần tham số 'with_noise=True', hãy thêm vào đây
    #             f_old = self._old_network.backbone(inputs).double()
    #             f_new = self._network.backbone(inputs).double()
                
    #             # Monitor drift thực tế trên batch này
    #             diff = (f_new - f_old).norm(p=2, dim=1).mean()
    #             diff_sum += diff.item()
    #             count += 1
                
    #             XtX += f_old.t() @ f_old
    #             XtY += f_old.t() @ f_new
        
    #     # Regularization (Ridge)
    #     reg = 0.1 * torch.eye(dim, device=self.device, dtype=torch.float64) 
    #     P = torch.linalg.solve(XtX + reg, XtY)
        
    #     # [CHECK POINT]: Kiểm tra xem P có phải Identity không
    #     P_float = P.float()
    #     identity = torch.eye(dim, device=self.device)
    #     deviation = torch.norm(P_float - identity)
        
    #     print("\n" + "="*40)
    #     print(f"--> [DPCR MONITOR] Task {self.cur_task}")
    #     print(f"    + Average Feature Shift Norm: {diff_sum/count:.4f}")
    #     print(f"    + P Matrix Deviation (||P-I||): {deviation:.4f}")
        
    #     if deviation < 0.01:
    #         print("    [CẢNH BÁO ĐỎ] ⚠️ Drift gần như bằng 0!")
    #         print("    Lý do: Backbone Frozen và Noise không thay đổi hoặc không được kích hoạt.")
    #         print("    Giải pháp: Kiểm tra lại hàm forward của backbone xem có cộng Noise không?")
    #     else:
    #         print("    [OK] ✅ DPCR đã phát hiện Semantic Shift. P Matrix hợp lệ.")
    #     print("="*40 + "\n")

    #     del XtX, XtY
    #     self._old_network.cpu() 
    #     self._clear_gpu()
        
    #     return P_float
    
    #thêm log
    def calculate_drift(self, clean_loader):
        """Tính Drift P & DEBUG xem Noise có hoạt động không"""
        print("\n" + "="*20 + " DEBUG DRIFT " + "="*20)
        
        # 1. Ép chuyển sang train() để kích hoạt Noise (nếu nó hoạt động như Dropout)
        self._network.train()
        self._old_network.train()
        
        # Đưa old_network lên GPU
        self._old_network.to(self.device)
        
        # [DEBUG 1]: Kiểm tra xem trọng số Noise của Mới và Cũ có khác nhau không?
        # Nếu diff = 0 nghĩa là quá trình train (run) vừa rồi vô dụng -> Cần xem lại optimizer
        noise_weight_diff = 0.0
        try:
            params_new = list(self._network.backbone.noise_maker.parameters())
            params_old = list(self._old_network.backbone.noise_maker.parameters())
            for p1, p2 in zip(params_new, params_old):
                noise_weight_diff += (p1 - p2).norm().item()
            print(f"--> [CHECK WEIGHT] Total Noise Param Diff (New - Old): {noise_weight_diff:.6f}")
        except:
            print("--> [CHECK WEIGHT] Không thể truy cập noise_maker parameters.")

        dim = self._network.feature_dim
        XtX = torch.zeros(dim, dim, device=self.device, dtype=torch.float64)
        XtY = torch.zeros(dim, dim, device=self.device, dtype=torch.float64)
        ref_dtype = next(self._network.backbone.parameters()).dtype
        
        diff_sum = 0.0
        count = 0
        
        # [DEBUG 2]: Kiểm tra feature đầu ra
        with torch.no_grad():
            for i, (idx, inputs, target) in enumerate(clean_loader):
                inputs = inputs.to(self.device, dtype=ref_dtype)
                
                # Forward qua backbone
                f_old = self._old_network.backbone(inputs).double()
                f_new = self._network.backbone(inputs).double()
                
                # Tính độ lệch feature trên batch này
                batch_diff = (f_new - f_old).norm(p=2, dim=1).mean().item()
                diff_sum += batch_diff
                count += 1
                
                # Chỉ in debug batch đầu tiên
                if i == 0:
                    print(f"--> [CHECK FEATURE] Batch 0 Feature Shift Norm: {batch_diff:.6f}")
                    if batch_diff == 0:
                        print("    [CẢNH BÁO] Feature Old và New giống hệt nhau!")
                        print("    -> Backbone frozen + Noise không có tác dụng.")
                    
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        # Regularization
        reg = 0.01 * torch.eye(dim, device=self.device, dtype=torch.float64) 
        P = torch.linalg.solve(XtX + reg, XtY)
        
        P_float = P.float()
        deviation = torch.norm(P_float - torch.eye(dim, device=self.device))
        
        print(f"--> [RESULT] Avg Feature Shift: {diff_sum/count:.6f}")
        print(f"--> [RESULT] P Matrix Deviation: {deviation:.4f}")
        print("="*55 + "\n")

        del XtX, XtY
        self._old_network.cpu() 
        self._clear_gpu()
        
        # Trả lại trạng thái eval
        self._network.eval()
        
        return P_float
    
    def fit_fc(self, train_loader):
        """
        Dùng hàm fit gốc (RLS) để train classifier.
        Dùng cho Task 0 và bước Warm-up ở các Task sau.
        """
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch), desc=f"RLS Fit Task {self.cur_task}")
        for _, epoch in enumerate(prog_bar):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets) # Gọi hàm fit RLS gốc
            
            self._clear_gpu()

    def re_fit(self, train_loader):
        """
        [DPCR Update]: TSSP -> CIP -> Ridge Regression -> CN
        """
        self._network.eval()
        self._network.to(self.device)

        # [CASE 1]: Task 0 - Chỉ cần lưu stats, ko cần DPCR
        if self.cur_task == 0:
            self.logger.info("--> Task 0: Collecting Backbone Statistics...")
            self._network._saved_mean = {}
            self._network._saved_cov = {}
            self._network._saved_count = {}
            
            # [FIX LỖI UNPACK]: Thêm ngoặc ( ) để unpack đúng
            for i, (_, inputs, targets) in enumerate(tqdm(train_loader, desc="Task 0 Stats")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Lưu stats 768d
                self._network.update_backbone_stats(inputs, targets)
            
            self._clear_gpu()
            return

        # [CASE 2]: Task > 0 - Full DPCR Pipeline
        # 1. Tính TSSP (Drift P) [cite: 167]
        P = self.calculate_drift(train_loader)
        boundary = self.known_class - self.increment
        
        # 2. CIP Replay: Sinh dữ liệu cũ đã bù Drift + Subspace [cite: 181]
        self.logger.info("--> DPCR: CIP Replay & Drift Correction...")
        HTH_old, HTY_old = self._network.solve_dpcr(P_drift=P, boundary=boundary)
        
        # 3. Gom dữ liệu mới
        HTH_curr = torch.zeros_like(HTH_old)
        
        current_total_class = self._network.known_class
        HTY_curr = torch.zeros(self.buffer_size, current_total_class, device=self.device)
        
        prog_bar = tqdm(train_loader, desc="DPCR: Collecting New Task")
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            y_oh = torch.nn.functional.one_hot(targets, num_classes=current_total_class).float()
            
            feat = self._network.buffer(self._network.backbone(inputs)).float()
            HTH_curr += feat.t() @ feat
            HTY_curr += feat.t() @ y_oh
            
            # Lưu stats backbone cho task này
            self._network.update_backbone_stats(inputs, targets)

        # Pad HTY_old nếu class tăng lên
        if HTY_old.shape[1] < HTY_curr.shape[1]:
            pad = torch.zeros(self.buffer_size, HTY_curr.shape[1] - HTY_old.shape[1], device=self.device)
            HTY_old = torch.cat([HTY_old, pad], dim=1)

        # 4. Giải Ridge Regression (Classifier Reconstruction) [cite: 236]
        HTH_total = HTH_old + HTH_curr
        HTY_total = HTY_old + HTY_curr
        W = self._network.simple_ridge_solve(HTH_total, HTY_total)
        
        # 5. Áp dụng Category-wise Normalization (CN) [cite: 252]
        W = self._network.category_normalization(W)
        
        # 6. Cập nhật Weight
        if self._network.weight.shape[1] < W.shape[1]:
             new_w = torch.zeros((self.buffer_size, W.shape[1]), device=self.device)
             new_w[:, :self._network.weight.shape[1]] = self._network.weight
             self._network.register_buffer("weight", new_w)
             
        self._network.weight.data = W
        
        del P, HTH_old, HTY_old, HTH_curr, HTY_curr, HTH_total, HTY_total
        self._clear_gpu()

    # (Các hàm khác giữ nguyên như cũ)
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
        scale = max(0.65, min(scale, 0.95))
        self.logger.info(f"--> [ADAPTIVE] Similarity: {max_sim:.4f} => Scale: {scale:.4f}")
        return scale

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
        
        current_scale = 0.85 
        if self.cur_task > 0:
            current_scale = self.compute_adaptive_scale(train_loader)

        # 1. Freeze / Unfreeze Logic
        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
        
        if self.cur_task == 0: 
            self._network.init_unfreeze()
        else: 
            self._network.unfreeze_noise()
            
        # [DEBUG] Kiểm tra tham số train
        params = list(filter(lambda p: p.requires_grad, self._network.parameters()))
        print(f"\n--> [DEBUG TASK {self.cur_task}] Params to train: {len(params)}")
        if len(params) == 0:
            print("❌ LỖI: Optimizer rỗng! Kiểm tra lại hàm unfreeze_noise.")
            return

        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        WARMUP_EPOCHS = 2

        for _, epoch in enumerate(prog_bar):
            losses, correct, total = 0.0, 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # [FIX 1] Xóa set_to_none=True để an toàn hơn khi debug
                optimizer.zero_grad() 
                
                # [FIX 2] TẮT AUTOCAST (Dùng FP32 thuần túy)
                # with autocast('cuda'): <--- ĐÃ BỎ
                if True:
                    if self.cur_task > 0:
                        with torch.no_grad():
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        # Forward pass
                        logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    
                    loss = F.cross_entropy(logits_final, targets.long())

                # [FIX 3] BACKWARD TRỰC TIẾP (Không dùng Scaler)
                loss.backward()
                
                # GPM Projection (nếu cần)
                if self.cur_task > 0:
                    if epoch >= WARMUP_EPOCHS:
                        # Không cần unscale vì đang dùng FP32
                        self._network.apply_gpm_to_grads(scale=0.85)
                
                # [DEBUG QUAN TRỌNG] Check Gradient Batch đầu tiên
                if self.cur_task > 0 and i == 0: 
                    total_norm = 0.0
                    for p in params:
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    print(f"    [Ep {epoch} Batch 0] Gradient Norm: {total_norm:.6f}")
                    if math.isnan(total_norm):
                        print("    ❌ LỖI CHẾT NGƯỜI: Gradient vẫn bị NaN!")
                        # Clip grad để chống crash tạm thời (nhưng cần sửa init)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                    elif total_norm == 0:
                        print("    ⚠️ CẢNH BÁO: Gradient = 0! Noise không học.")

                # [FIX 4] STEP TRỰC TIẾP
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                # Clean memory
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f} | Scale {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, current_scale
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            
            if epoch % 5 == 0: self._clear_gpu()
    
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