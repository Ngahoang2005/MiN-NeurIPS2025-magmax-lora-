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
        
        self.scaler = GradScaler('cuda')

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def after_train(self, data_manger):
        # Logic cập nhật known_class được di chuyển xuống cuối hàm này 
        # để đảm bảo eval_task tính đúng Acc Mới/Cũ cho task vừa train xong
        
        # [MODIFIED] Eval Accuracy
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        
        # Gọi eval_task trước khi update self.known_class để phân biệt cũ/mới
        eval_res = self.eval_task(test_loader)
        
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info(f"Old Acc: {eval_res.get('old_class_accy', 0):.2f} | New Acc: {eval_res.get('new_class_accy', 0):.2f}")
        
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        print(f"Old Acc: {eval_res.get('old_class_accy', 0):.2f} | New Acc: {eval_res.get('new_class_accy', 0):.2f}")

        # Update known_class cho task tiếp theo
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment
            
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
        
        self._clear_gpu()
        
        # Sử dụng buffer_batch cho RLS
        train_loader_rls = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader_rls = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        
        self.fit_fc(train_loader_rls, test_loader_rls)

        # [NEW] Tính toán thống kê Mean/Cov sau khi train xong Task 0
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_stats = DataLoader(train_set_no_aug, batch_size=self.init_batch_size, shuffle=False,
                                        num_workers=self.num_workers)
        self._network.compute_class_statistics(train_loader_stats)
        
        del train_set, test_set, train_set_no_aug
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

        # Loader cho RLS (fit_fc)
        train_loader_rls = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader_rls = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)

        self.test_loader = test_loader_rls

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        # [MODIFIED] Fit RLS với Pseudo Replay
        self.fit_fc(train_loader_rls, test_loader_rls)

        self._network.update_fc(self.increment)

        # Loader cho Gradient Descent (Noise Training)
        train_loader_gd = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        
        self._clear_gpu()

        self.run(train_loader_gd)
        self._network.collect_projections(mode='threshold', val=0.95)
        
        self._clear_gpu()
        
        # [NEW] Tính toán thống kê Mean/Cov cho Task hiện tại
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_stats = DataLoader(train_set_no_aug, batch_size=self.init_batch_size, shuffle=False,
                                        num_workers=self.num_workers)
        self._network.compute_class_statistics(train_loader_stats)

        del train_set, train_set_no_aug

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # Ngưỡng an toàn cho Backbone Forward (tùy GPU, 64-128 thường ổn với ViT-B)
        BACKBONE_CHUNK_SIZE = 64 

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs kích thước lớn (ví dụ 512)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # [FIX OOM] 1. Extract Features theo từng chunk nhỏ
                real_features_list = []
                
                # Tắt Autocast hoàn toàn để tránh cache memory không cần thiết
                # và dùng torch.no_grad() cho backbone để tiết kiệm VRAM tối đa
                with torch.no_grad(): 
                    # Chia nhỏ inputs thành các chunks
                    input_chunks = torch.split(inputs, BACKBONE_CHUNK_SIZE)
                    
                    for chunk in input_chunks:
                        # Forward chunk qua backbone
                        # Ép kiểu float() ngay sau khi output để giải phóng graph (dù no_grad ko tạo graph nhưng cho chắc)
                        chunk_feat = self._network.backbone(chunk).float()
                        real_features_list.append(chunk_feat)
                
                # Gộp lại thành batch lớn như cũ
                real_features = torch.cat(real_features_list, dim=0)
                
                # [Replay Logic]
                with autocast('cuda', enabled=False):
                    pseudo_features, pseudo_labels = None, None
                    if self.cur_task > 0:
                        num_pseudo = max(1, real_features.shape[0] // 4)
                        p_x, p_y = self._network.generate_pseudo_features(num_samples_per_class=1)
                        if p_x is not None:
                            pseudo_features = p_x.float()
                            pseudo_labels = p_y

                    # Combine Real + Pseudo
                    if pseudo_features is not None:
                        combined_features = torch.cat([real_features, pseudo_features], dim=0)
                        combined_targets = torch.cat([targets, pseudo_labels], dim=0)
                    else:
                        combined_features = real_features
                        combined_targets = targets
                    
                    # One-hot encoding
                    # Lưu ý: Cần đảm bảo weight đã đủ lớn
                    current_dim = self._network.weight.shape[1]
                    max_label = combined_targets.max().item()
                    if max_label >= current_dim:
                         # Nếu label mới lớn hơn dim hiện tại của weight, cần expand logic bên ngoài 
                         # hoặc dùng max_label + 1. 
                         # Tuy nhiên fit() thường tự xử lý. Ở đây ta làm thủ công cho RLS:
                         req_dim = max_label + 1
                         if req_dim > current_dim:
                             inc = req_dim - current_dim
                             tail = torch.zeros((self._network.weight.shape[0], inc), device=self.device)
                             self._network.weight = torch.cat((self._network.weight, tail), dim=1)
                             current_dim = req_dim

                    Y_onehot = F.one_hot(combined_targets.long(), num_classes=current_dim).float()
                    
                    # 2. RLS Update (Trên toàn bộ batch lớn)
                    # Pass qua Buffer (Random Projection)
                    X_proj = self._network.buffer(combined_features)
                    
                    # Dọn dẹp features gốc để tiết kiệm VRAM trước khi tính toán ma trận
                    del real_features, combined_features, pseudo_features, real_features_list
                    torch.cuda.empty_cache()

                    X_proj, Y_onehot = X_proj.to(self.device), Y_onehot.to(self.device)
                    
                    # Công thức RLS (Recursive Least Squares)
                    # P = (I + X R X^T)^-1
                    term = torch.eye(X_proj.shape[0], device=self.device) + X_proj @ self._network.R @ X_proj.T
                    jitter = 1e-6 * torch.eye(term.shape[0], device=self.device)
                    
                    try:
                        # K = R X^T (X R X^T + I)^-1
                        # Dùng linalg.solve cho ổn định: (Term) K_T = (X R) -> K = K_T.T
                        K = torch.linalg.solve(term + jitter, X_proj @ self._network.R)
                        K = K.T
                    except:
                        K = self._network.R @ X_proj.T @ torch.inverse(term + jitter)
                        
                    self._network.R -= K @ X_proj @ self._network.R
                    self._network.weight += K @ (Y_onehot - X_proj @ self._network.weight)
                    
                    del term, jitter, K, X_proj, Y_onehot

            info = "Task {} --> Update Analytical Classifier (w/ Chunking)!".format(
                self.cur_task,
            )
            # self.logger.info(info) # Log ít lại cho đỡ spam
            prog_bar.set_description(info)
            self._clear_gpu()
    def re_fit(self, train_loader, test_loader):
        # Tương tự fit_fc nhưng không cần Replay quá nhiều nếu chỉ là refinement
        # Hoặc copy logic fit_fc xuống đây.
        self.fit_fc(train_loader, test_loader) 

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
                        logits_final = logits2 + logits1
                    else:
                        logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    
                    loss = F.cross_entropy(logits_final, targets.long())

                self.scaler.scale(loss).backward()
                
                if self.cur_task > 0:
                    if epoch >= WARMUP_EPOCHS:
                        self.scaler.unscale_(optimizer)
                        self._network.apply_gpm_to_grads(scale=current_scale)
                    else:
                        pass
                
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
            scheduler.step()
            train_acc = 100. * correct / total

            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            
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
        
        # [NEW] Tính Acc cho Old và New Class
        pred_np = np.array(pred)
        label_np = np.array(label)
        
        # self.known_class là số lượng class CŨ (trước khi học task hiện tại)
        # Nếu đang ở after_train của Task T, self.known_class vẫn đang trỏ tới đầu Task T
        # Tức là: < self.known_class là Cũ, >= self.known_class là Mới
        
        old_mask = label_np < self.known_class
        new_mask = label_np >= self.known_class
        
        old_acc = 0.0
        new_acc = 0.0
        
        if np.sum(old_mask) > 0:
            old_acc = (pred_np[old_mask] == label_np[old_mask]).mean() * 100.
            
        if np.sum(new_mask) > 0:
            new_acc = (pred_np[new_mask] == label_np[new_mask]).mean() * 100.
        
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
            "old_class_accy": old_acc, # Thêm trường này
            "new_class_accy": new_acc  # Thêm trường này
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