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
        
        # 1. RUN
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # 2. FIT CLASSIFIER
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader_buf = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.fit_fc(train_loader_buf, test_loader_buf)

        # 3. RE-FIT & SAVE STATS (Quan trọng cho Task 0)
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(train_loader_no_aug, test_loader_buf)
        
        del train_set, test_set, train_set_no_aug
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self.test_loader = test_loader
        
        # [SAVE OLD MODEL] cho tính drift
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        # 1. Fit classifier teacher
        self.fit_fc(train_loader, test_loader)

        # 2. Update network
        self._network.update_fc(self.increment)

        # 3. Train Backbone (RUN)
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()
        self.run(train_loader_run)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()

        del train_loader_run

        # 4. RE-FIT & DPCR
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_no_aug, test_loader)
        
        del train_set, test_set, train_set_no_aug
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        """RLS thuần túy để train classifier mạnh nhất cho task hiện tại"""
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)
            self._clear_gpu()

    def calculate_drift(self, clean_loader):
        """Tính Drift P trên Backbone Space (768-d)"""
        self._network.eval()
        self._old_network.eval()
        self._network.to(self.device)
        self._old_network.to(self.device)
        
        dim = self._network.feature_dim
        XtX = torch.zeros(dim, dim, device=self.device)
        XtY = torch.zeros(dim, dim, device=self.device)
        ref_dtype = next(self._network.backbone.parameters()).dtype
        
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                inputs = inputs.to(self.device, dtype=ref_dtype)
                f_old = self._old_network.backbone(inputs).float()
                f_new = self._network.backbone(inputs).float()
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        # Regularization mạnh hơn chút (0.1) cho backbone space
        reg = 0.1 * torch.eye(dim, device=self.device) 
        P = torch.linalg.solve(XtX + reg, XtY)
        
        del XtX, XtY
        self._clear_gpu()
        return P

    def re_fit(self, train_loader, test_loader):
        """Quy trình Backbone Space DPCR"""
        self._network.eval()
        self._network.to(self.device)

        # [CASE 1]: Task 0 - Chỉ lưu stats
        if self.cur_task == 0:
            self.logger.info("--> Task 0: Saving Backbone Statistics...")
            self._network._saved_mean = {}
            self._network._saved_cov = {}
            self._network._saved_count = {}
            
            for i, (_, inputs, targets) in enumerate(tqdm(train_loader, desc="Collecting Task 0 Stats")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Lưu stats 768d
                self._network.update_backbone_stats(inputs, targets)
                # Fit nhẹ lại RLS để đồng bộ tối đa (optional)
                y_oh = torch.nn.functional.one_hot(targets)
                self._network.fit(inputs, y_oh)
                
            self._clear_gpu()
            return

        # [CASE 2]: Task > 0 - Replay + Drift
        P = self.calculate_drift(train_loader)
        boundary = self.known_class - self.increment
        
        # 1. Sinh Replay Data từ stats cũ (đã bù drift)
        self.logger.info("--> Generating Replay Features...")
        HTH_old, HTY_old = self._network.solve_using_backbone_stats(P_drift=P, boundary=boundary)
        
        # 2. Gom dữ liệu thực tế Task hiện tại
        HTH_curr = torch.zeros_like(HTH_old)
        
        # Init HTY_curr đủ lớn để chứa class mới
        # (solve_using_backbone_stats đã trả về HTY_old size [Buffer, Old_Classes])
        # Ta cần mở rộng nó ra
        
        # Biến tạm để xác định max class id
        max_cls = 0 
        
        for _, inputs, targets in tqdm(train_loader, desc="Collecting Current Task"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            y_oh = torch.nn.functional.one_hot(targets)
            
            # Tính toán cho giải hệ phương trình (Qua Buffer)
            feat = self._network.buffer(self._network.backbone(inputs)).float()
            HTH_curr += feat.t() @ feat
            
            # Xử lý HTY (Dynamic Size)
            # Nếu HTY_curr chưa khởi tạo đúng size
            current_batch_max = y_oh.shape[1]
            max_cls = max(max_cls, current_batch_max)
            
            # Pad nếu cần thiết để cộng dồn
            if not isinstance(HTY_curr, torch.Tensor) or HTY_curr.sum() == 0: # First init
                 HTY_curr = torch.zeros(self.buffer_size, current_batch_max, device=self.device)
            
            if current_batch_max > HTY_curr.shape[1]:
                pad = torch.zeros(self.buffer_size, current_batch_max - HTY_curr.shape[1], device=self.device)
                HTY_curr = torch.cat([HTY_curr, pad], dim=1)
            
            if HTY_curr.shape[1] > y_oh.shape[1]:
                pad = torch.zeros(y_oh.shape[0], HTY_curr.shape[1] - y_oh.shape[1], device=self.device)
                y_oh = torch.cat([y_oh, pad], dim=1)
                
            HTY_curr += feat.t() @ y_oh
            
            # Lưu Stats Backbone cho Task này (để dùng cho tương lai)
            self._network.update_backbone_stats(inputs, targets)

        # 3. Tổng hợp và Giải
        # Pad HTY_old và HTY_curr cho cùng kích thước
        total_cols = max(HTY_old.shape[1], HTY_curr.shape[1])
        if HTY_old.shape[1] < total_cols:
            pad = torch.zeros(self.buffer_size, total_cols - HTY_old.shape[1], device=self.device)
            HTY_old = torch.cat([HTY_old, pad], dim=1)
        if HTY_curr.shape[1] < total_cols:
            pad = torch.zeros(self.buffer_size, total_cols - HTY_curr.shape[1], device=self.device)
            HTY_curr = torch.cat([HTY_curr, pad], dim=1)

        HTH_total = HTH_old + HTH_curr
        HTY_total = HTY_old + HTY_curr
        
        reg = self.gamma * torch.eye(self.buffer_size, device=self.device)
        W = torch.linalg.solve(HTH_total + reg, HTY_total)
        
        # Cập nhật Weight (Resize nếu cần)
        if self._network.weight.shape[1] < W.shape[1]:
             new_w = torch.zeros((self.buffer_size, W.shape[1]), device=self.device)
             new_w[:, :self._network.weight.shape[1]] = self._network.weight
             self._network.register_buffer("weight", new_w)
             
        self._network.weight.data = W
        
        del P, HTH_old, HTY_old, HTH_curr, HTY_curr
        self._clear_gpu()

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
            losses, correct, total = 0.0, 0, 0
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
                        self._network.apply_gpm_to_grads(scale=0.85)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f} | Scale {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, current_scale
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            print(info)
            if epoch % 5 == 0: self._clear_gpu()

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