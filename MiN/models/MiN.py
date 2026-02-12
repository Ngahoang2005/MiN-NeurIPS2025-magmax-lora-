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
            self.analyze_cosine_accuracy(test_loader)  # [ADDED] Phân tích mối quan hệ giữa Cosine Similarity và Accuracy sau mỗi task
        del test_set
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
        
        # [FIX OOM] Dọn GPU trước và sau khi tính proto
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        self._network.set_noise_mode(-2)
        self.run(train_loader)
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
                                 num_workers=self.num_workers)
        self._network.set_noise_mode(-2)
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
        
        # [ADDED] Clear memory
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

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.fit_fc(train_loader, test_loader)

        self._network.update_fc(self.increment)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        
        self.run(train_loader)
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)

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
                # Logic gốc: Fit Analytical (RLS). 
                # Không dùng Autocast ở đây vì RLS cần độ chính xác cao (ma trận nghịch đảo)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(
                self.cur_task,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            # [ADDED] Clear cache sau mỗi epoch fit
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets)
            self._network.fit(inputs, targets)

            info = "Task {} --> Reupdate Analytical Classifier!".format(
                self.cur_task,
            )
            
            self.logger.info(info)
            prog_bar.set_description(info)
        self._clear_gpu()

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True
            
        if self.cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        prog_bar = tqdm(range(epochs))
        self._network.train()
        self._network.to(self.device)
        self._network.set_noise_mode(-2)
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # [ADDED] set_to_none=True tiết kiệm RAM hơn
                optimizer.zero_grad(set_to_none=True) 

                # [ADDED] Autocast để giảm 50% VRAM khi train
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits2 = logits2 + logits1
                        loss = F.cross_entropy(logits2, targets.long())
                        logits_final = logits2
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets.long())
                        logits_final = logits

                # [ADDED] Backward với Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()

                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                # [ADDED] Xóa biến tạm
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total

            info = "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            self.logger.info(info)
            prog_bar.set_description(info)
            
            # [ADDED] Clear cache sau mỗi epoch
            if epoch % 5 == 0:
                self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                # [MODIFIED] Logic Selection
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
    # =========================================================================
    # [FIX OOM] HÀM NÀY ĐÃ ĐƯỢC CHỈNH ĐỂ CHẠY TRÊN CPU
    # Vẫn giữ nguyên logic là Simple Mean (Mean tất cả feature)
    # =========================================================================
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
            decay = 0.9 
            self._network.H *= decay
            self._network.Hy *= decay
        else:
            self._network.H.zero_()
            self._network.Hy.zero_()

        print(f">>> [Fast RLS] Accumulating Stats for Task {self.cur_task}...")
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit_batch(inputs, targets)
            
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

    def analyze_cosine_accuracy(self, test_loader):
        """Vẽ biểu đồ: Mối quan hệ giữa Cosine Similarity và Accuracy"""
        self._network.eval()
        all_similarities = []
        all_correct_flags = []
    
        print(f">>> [DEBUG] Analyzing Cosine Similarity vs Accuracy for Task {self.cur_task}...")
        
        # Lấy prototype của task hiện tại để so sánh
        current_proto = self._network.task_prototypes[self.cur_task].to(self.device)
        current_proto = F.normalize(current_proto.unsqueeze(0), p=2, dim=1)

        with torch.no_grad():
            for _, inputs, targets in tqdm(test_loader, desc="Collecting Cosine Data"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Lấy feature ở mode Universal
                self._network.set_noise_mode(-2)
                feat = self._network.extract_feature(inputs)
                feat_norm = F.normalize(feat, p=2, dim=1)
                
                # Tính Sim với Prototype của task đúng
                sim = torch.mm(feat_norm, current_proto.t()).squeeze(1)
                
                # Lấy dự đoán cuối cùng (dùng hàm combined chuẩn)
                outputs = self._network.forward_tuna_combined(inputs)
                predicts = torch.max(outputs['logits'], dim=1)[1]
                correct = (predicts == targets).float()

                all_similarities.extend(sim.cpu().numpy())
                all_correct_flags.extend(correct.cpu().numpy())

        self._plot_cosine_graph(all_similarities, all_correct_flags)

    def _plot_cosine_graph(self, sims, corrects, num_bins=10):
        try:
            sims = np.array(sims)
            corrects = np.array(corrects)
            
            # Chia bins theo độ tương đồng (0.0 đến 1.0)
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_accs = []
            bin_centers = []

            for i in range(num_bins):
                mask = (sims >= bin_edges[i]) & (sims < bin_edges[i+1])
                if mask.any():
                    bin_accs.append(corrects[mask].mean() * 100)
                    bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

            plt.figure(figsize=(8, 6))
            plt.bar(bin_centers, bin_accs, width=0.08, color='green', alpha=0.7, edgecolor='black')
            plt.plot(bin_centers, bin_accs, marker='o', color='darkgreen', linewidth=2)
            
            plt.xlabel('Cosine Similarity to Task Prototype', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'Reliability Diagram: Similarity vs Acc (Task {self.cur_task})', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.ylim(0, 105)
            
            plt.savefig(f'debug_cosine_acc_task_{self.cur_task}.png')
            plt.close()
            print(f">>> [PLOT] Saved Cosine Debug Chart: debug_cosine_acc_task_{self.cur_task}.png")
        except Exception as e:
            print(f">>> [PLOT ERROR] {e}")