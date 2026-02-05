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
        self._old_network = None # [FIX]: Đã thêm lại biến này
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
        self._network.collect_projections(mode='threshold', val=0.95)
        #self._network.after_task_magmax_merge()
        #self.analyze_model_sparsity()
        
        self._clear_gpu()
        
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
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
        
        # nén
        self.logger.info("--> [DPCR] Compressing Task 0 statistics...")
        self._network.temp_phi = {} 
        self._network.temp_mu = {}
        self._network.temp_count = {}
        
        for _, inputs, targets in tqdm(train_loader, desc="Task 0 Compression"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            y_oh = torch.nn.functional.one_hot(targets)
            self._network.accumulate_stats(inputs, y_oh)
            
        self._network.compress_stats()
        self._clear_gpu()
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
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.fit_fc(train_loader, test_loader)

        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        
        self._clear_gpu()

        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
       
        self._clear_gpu()


        del train_set

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

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(
                self.cur_task,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        # [CASE 1]: Task 0 -> Dùng RLS gốc (Acc 99.3%)
        # Task 0 không cần giải DPCR ở đây, việc nén Task 0 đã làm ở init_train rồi
        if self.cur_task == 0:
            prog_bar = tqdm(train_loader, desc="RLS Refitting (Task 0)")
            for i, (_, inputs, targets) in enumerate(prog_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets) # Gọi hàm fit gốc RLS
            self._clear_gpu()
            return

        # [CASE 2]: Task > 0 -> Quy trình DPCR chuẩn
        # Tính toán Drift từ Model cũ (đã lưu ở đầu increment_train)
        P = self.calculate_drift(train_loader)
        boundary = self.known_class - self.increment
        
        # Reset biến tạm để gom dữ liệu Task này
        self._network.temp_phi = {}
        self._network.temp_mu = {}
        self._network.temp_count = {}

        # 1. GOM DỮ LIỆU (Accumulate)
        for _, inputs, targets in tqdm(train_loader, desc="DPCR Accumulating"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            y_oh = torch.nn.functional.one_hot(targets)
            self._network.accumulate_stats(inputs, y_oh)
        
        # 2. GIẢI HỆ PHƯƠNG TRÌNH (Solve)
        # Giải TRƯỚC KHI NÉN để tận dụng dữ liệu thô (temp_phi) của Task hiện tại -> Acc cao hơn
        self.logger.info(f"--> Solving DPCR with Drift Correction (Boundary: {boundary})...")
        self._network.solve_analytic(P_drift=P, boundary=boundary)

        # 3. NÉN DỮ LIỆU (Compress)
        # Giờ mới nén temp_phi vào compressed_stats để dành cho Task sau
        self._network.compress_stats()
        self._network.temp_phi = {} 
        self._network.temp_mu = {}
        
       
        
        self._clear_gpu()
    def compute_adaptive_scale(self, current_loader):
        # 1. Tính prototype task hiện tại
        curr_proto = self.get_task_prototype(self._network, current_loader)
        
        if not hasattr(self, 'old_prototypes'): self.old_prototypes = []
        
        if not self.old_prototypes:
            self.old_prototypes.append(curr_proto)
            return 0.95 # Task đầu tiên chưa cần scale, hoặc scale cao
            
        # 2. So sánh với quá khứ
        max_sim = 0.0
        curr_norm = F.normalize(curr_proto.unsqueeze(0), p=2, dim=1)
        for old_p in self.old_prototypes:
            old_norm = F.normalize(old_p.unsqueeze(0), p=2, dim=1)
            sim = torch.mm(curr_norm, old_norm.t()).item()
            if sim > max_sim: max_sim = sim
                
        self.old_prototypes.append(curr_proto)
        
        # 3. Tính Scale: Giống nhau nhiều -> Scale thấp (để học đè lên). Khác nhau -> Scale cao.
        # Công thức: Scale chạy từ 0.5 đến 0.95
        scale = 0.5 + 0.5 * (1.0 - max_sim)
        scale = max(0.65, min(scale, 0.95)) # Kẹp giá trị an toàn
        
        self.logger.info(f"--> [ADAPTIVE] Similarity: {max_sim:.4f} => Scale: {scale:.4f}")
        return scale
    def run(self, train_loader):
        # [TỐI ƯU 1]: Import nên để đầu file, nhưng nếu để đây cũng ko sao.
        # scaler = GradScaler() -> [SAI]: Đừng tạo mới mỗi task!
        # Hãy dùng self.scaler đã tạo trong __init__

        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

        # [TỐI ƯU 2]: Tính scale một lần đầu task
        current_scale = 0.85 
        if self.cur_task > 0:
            # Đảm bảo bạn đã thêm hàm compute_adaptive_scale vào class MinNet nhé
            current_scale = self.compute_adaptive_scale(train_loader)

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

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 

                # Tự động detect device cho autocast
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            # forward cũ
                            logits1 = self._network(inputs, new_forward=False)['logits']
                        # forward mới
                        logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    
                    loss = F.cross_entropy(logits_final, targets.long())

                # [TỐI ƯU 3]: Dùng self.scaler
                self.scaler.scale(loss).backward()
                
                # Logic GPM + Warmup
                if self.cur_task > 0:
                    # Đã vào task > 0 thì check epoch thôi
                    if epoch >= WARMUP_EPOCHS:
                        self.scaler.unscale_(optimizer)
                        # Áp dụng Adaptive Scale
                        self._network.apply_gpm_to_grads(scale=0.85)
                    else:
                        # Warm-up: Thả trôi gradient để học nhanh
                        pass
                
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                # Cleanup
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total

            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f} | Scale {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, current_scale
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            print(info)
            
            # Clear cache định kỳ
            if epoch % 5 == 0:
                self._clear_gpu()
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
    def calculate_drift(self, clean_loader):
        """
        Tính toán ma trận chuyển đổi P để bù đắp sự thay đổi (Drift) 
        của backbone sau khi train task mới.
        """
        self._network.eval()
        self._old_network.eval()
        
        # Đảm bảo cả 2 network nằm trên đúng thiết bị
        self._network.to(self.device)
        self._old_network.to(self.device)
        
        bs = self.buffer_size
        # Khởi tạo ma trận hiệp phương sai và ma trận tương quan chéo trên GPU (FP32)
        XtX = torch.zeros(bs, bs, device=self.device, dtype=torch.float32)
        XtY = torch.zeros(bs, bs, device=self.device, dtype=torch.float32)
        
        # Lấy dtype chuẩn của backbone để tránh lỗi mismatch
        ref_dtype = next(self._network.backbone.parameters()).dtype
        
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                inputs = inputs.to(self.device, dtype=ref_dtype)
                
                # Trích xuất đặc trưng từ model cũ và model mới
                # Ép về float() ngay sau buffer để tính toán ma trận chính xác
                f_old = self._old_network.buffer(self._old_network.backbone(inputs)).float()
                f_new = self._network.buffer(self._network.backbone(inputs)).float()
                
                # Tích lũy ma trận: P = (f_old^T * f_old)^-1 * (f_old^T * f_new)
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        # Giải hệ phương trình tuyến tính để tìm P: XtX * P = XtY
        # Thêm một lượng epsilon nhỏ (reg) để ma trận không bị suy biến (singular)
        reg = 1e-4 * torch.eye(bs, device=self.device)
        P = torch.linalg.solve(XtX + reg, XtY)
        
        # Dọn dẹp bộ nhớ đệm
        del XtX, XtY, reg
        self._clear_gpu()
        
        return P
    