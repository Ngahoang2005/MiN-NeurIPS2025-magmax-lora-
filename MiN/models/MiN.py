# import math
# import random
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch import optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import copy
# import gc 
# import os

# from utils.inc_net import MiNbaseNet
# from utils.toolkit import tensor2numpy
# from data_process.data_manger import DataManger
# from utils.training_tool import get_optimizer, get_scheduler
# from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# try:
#     from torch.amp import autocast, GradScaler
# except ImportError:
#     from torch.cuda.amp import autocast, GradScaler

# EPSILON = 1e-8

# class MinNet(object):
#     def __init__(self, args, loger):
#         super().__init__()
#         self.args = args
#         self.logger = loger
#         self._network = MiNbaseNet(args)
#         self.device = args['device']
#         self.num_workers = args["num_workers"]

#         self.init_epochs = args["init_epochs"]
#         self.init_lr = args["init_lr"]
#         self.init_weight_decay = args["init_weight_decay"]
#         self.init_batch_size = args["init_batch_size"]

#         self.lr = args["lr"]
#         self.batch_size = args["batch_size"]
#         self.weight_decay = args["weight_decay"]
#         self.epochs = args["epochs"]

#         self.init_class = args["init_class"]
#         self.increment = args["increment"]

#         self.buffer_size = args["buffer_size"]
#         self.buffer_batch = args["buffer_batch"]
#         self.gamma = args['gamma']
#         self.fit_epoch = args["fit_epochs"]

#         self.known_class = 0
#         self.cur_task = -1
#         self.total_acc = []
#         self.class_acc = []
#         self.task_acc = []
        
#         self.scaler = GradScaler('cuda')

#     def _clear_gpu(self):
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def after_train(self, data_manger):
#         if self.cur_task == 0:
#             self.known_class = self.init_class
#         else:
#             self.known_class += self.increment

#         _, test_list, _ = data_manger.get_task_list(self.cur_task)
#         test_set = data_manger.get_task_data(source="test", class_list=test_list)
#         test_set.labels = self.cat2order(test_set.labels, data_manger)
#         test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
#                                  num_workers=self.num_workers)
        
#         # [EVAL] Kết hợp FeCAM (beta=0.6)
#         eval_res = self.eval_task(test_loader)
        
#         self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
#         self.logger.info('total acc: {}'.format(self.total_acc))
#         self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
#         self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
#         print('total acc: {}'.format(self.total_acc))
#         print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        
#         del test_set

#     def save_check_point(self, path_name):
#         torch.save(self._network.state_dict(), path_name)

#     @staticmethod
#     def cat2order(targets, datamanger):
#         for i in range(len(targets)):
#             targets[i] = datamanger.map_cat2order(targets[i])
#         return targets

#     def init_train(self, data_manger):
#         self.cur_task += 1
#         train_list, test_list, train_list_name = data_manger.get_task_list(0)
#         self.logger.info("task_list: {}".format(train_list_name))
#         self.logger.info("task_order: {}".format(train_list))

#         train_set = data_manger.get_task_data(source="train", class_list=train_list)
#         train_set.labels = self.cat2order(train_set.labels, data_manger)
#         test_set = data_manger.get_task_data(source="test", class_list=test_list)
#         test_set.labels = self.cat2order(test_set.labels, data_manger)

#         train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True,
#                                   num_workers=self.num_workers)
#         test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
#                                  num_workers=self.num_workers)

#         self.test_loader = test_loader

#         if self.args['pretrained']:
#             for param in self._network.backbone.parameters():
#                 param.requires_grad = True

#         self._network.update_fc(self.init_class)
#         self._network.update_noise()
        
#         self._clear_gpu()
        
#         self.run(train_loader)
#         self._network.collect_projections(mode='threshold', val=0.95)
#         #self._network.after_task_magmax_merge()
#         self._clear_gpu()
        
#         # Analytic Learning
#         train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
#                                   num_workers=self.num_workers)
#         test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
#                                  num_workers=self.num_workers)
#         self.fit_fc(train_loader, test_loader)

#         train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
#         train_set.labels = self.cat2order(train_set.labels, data_manger)
#         train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
#                                   num_workers=self.num_workers)
#         test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
#                                  num_workers=self.num_workers)

#         if self.args['pretrained']:
#             for param in self._network.backbone.parameters():
#                 param.requires_grad = False

#         self.re_fit(train_loader, test_loader)
       
#         del train_set, test_set
#         self._clear_gpu()

#     def increment_train(self, data_manger):
#         self.cur_task += 1
        
#         # [REMOVED]: Không tạo snapshot nữa để tránh OOM và vì đã bỏ đo Drift
        
#         train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
#         self.logger.info("task_list: {}".format(train_list_name))
#         self.logger.info("task_order: {}".format(train_list))

#         train_set = data_manger.get_task_data(source="train", class_list=train_list)
#         train_set.labels = self.cat2order(train_set.labels, data_manger)
#         test_set = data_manger.get_task_data(source="test", class_list=test_list)
#         test_set.labels = self.cat2order(test_set.labels, data_manger)

#         train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
#                                   num_workers=self.num_workers)
#         test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
#                                  num_workers=self.num_workers)

#         self.test_loader = test_loader

#         if self.args['pretrained']:
#             for param in self._network.backbone.parameters():
#                 param.requires_grad = False

#         self._network.update_fc(self.increment)

#         self.fit_fc(train_loader, test_loader)

        

#         train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
#                                     num_workers=self.num_workers)
#         self._network.update_noise()
        
#         self._clear_gpu()
        
#         self.run(train_loader)
        
#         # GPM Collect
#         self._network.collect_projections(mode='threshold', val=0.95)
#         #self._network.after_task_magmax_merge()
#         self._clear_gpu()

#         del train_set

#         train_set = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
#         train_set.labels = self.cat2order(train_set.labels, data_manger)

#         train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
#                                     num_workers=self.num_workers)
#         test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False,
#                                     num_workers=self.num_workers)

#         if self.args['pretrained']:
#             for param in self._network.backbone.parameters():
#                 param.requires_grad = False

#         self.re_fit(train_loader, test_loader)
#         self._network.weight_merging(alpha=0.1)
#         del train_set, test_set
#         self._clear_gpu()

#     def fit_fc(self, train_loader, test_loader):
#         self._network.eval()
#         self._network.to(self.device)

#         prog_bar = tqdm(range(self.fit_epoch))
#         for _, epoch in enumerate(prog_bar):
#             self._network.to(self.device)
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 targets = torch.nn.functional.one_hot(targets)
#                 self._network.fit(inputs, targets)
            
#             info = "Task {} --> Update Analytical Classifier!".format(
#                 self.cur_task,
#             )
#             self.logger.info(info)
#             prog_bar.set_description(info)
#             self._clear_gpu()

#     def re_fit(self, train_loader, test_loader):
#         self._network.eval()
#         self._network.to(self.device)
#         prog_bar = tqdm(train_loader)
#         for i, (_, inputs, targets) in enumerate(prog_bar):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             targets = torch.nn.functional.one_hot(targets)
#             self._network.fit(inputs, targets)

#             info = "Task {} --> Reupdate Analytical Classifier!".format(
#                 self.cur_task,
#             )
            
#             self.logger.info(info)
#             prog_bar.set_description(info)
#         self._clear_gpu()

#     def run(self, train_loader):
#         epochs = self.init_epochs if self.cur_task == 0 else self.epochs
#         lr = self.init_lr if self.cur_task == 0 else self.lr
#         weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

#         # Hardcoded scale (đã bỏ adaptive)
#         current_scale = 0.85

#         for param in self._network.parameters(): param.requires_grad = False
#         for param in self._network.normal_fc.parameters(): param.requires_grad = True
        
#         if self.cur_task == 0: self._network.init_unfreeze()
#         else: self._network.unfreeze_noise()
            
#         params = filter(lambda p: p.requires_grad, self._network.parameters())
#         optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
#         scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

#         prog_bar = tqdm(range(epochs))
#         self._network.train()
#         self._network.to(self.device)

#         WARMUP_EPOCHS = 5

#         for _, epoch in enumerate(prog_bar):
#             losses = 0.0
#             correct, total = 0, 0

#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 optimizer.zero_grad(set_to_none=True) 

#                 with autocast('cuda'):
#                     if self.cur_task > 0:
#                         with torch.no_grad():
#                             logits1 = self._network(inputs, new_forward=False)['logits'].detach()
#                         logits2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
#                         logits_final = logits2 + logits1
#                     else:
#                         logits_final = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    
#                     loss = F.cross_entropy(logits_final, targets.long())

#                 self.scaler.scale(loss).backward()
                
#                 if self.cur_task > 0:
#                     if epoch >= WARMUP_EPOCHS:
#                         self.scaler.unscale_(optimizer)
#                         self._network.apply_gpm_to_grads(scale=current_scale)
                
#                 self.scaler.step(optimizer)
#                 self.scaler.update()
                
#                 losses += loss.item()
#                 _, preds = torch.max(logits_final, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)
                
#                 del inputs, targets, loss, logits_final

#             scheduler.step()
#             train_acc = 100. * correct / total

#             info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f} | Scale {:.2f}".format(
#                 self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, current_scale
#             )
#             self.logger.info(info)
#             prog_bar.set_description(info)
#             print(info)
            
#             if epoch % 5 == 0:
#                 self._clear_gpu()

#     def eval_task(self, test_loader):
#         model = self._network.eval()
#         pred, label = [], []
#         with torch.no_grad():
#             for i, (_, inputs, targets) in enumerate(test_loader):
#                 inputs = inputs.to(self.device)
                
#                 outputs = model(inputs)
#                 logits = outputs["logits"]
#                 predicts = torch.max(logits, dim=1)[1]
#                 pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
#                 label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        
#         class_info = calculate_class_metrics(pred, label)
#         task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
#         return {
#             "all_class_accy": class_info['all_accy'],
#             "class_accy": class_info['class_accy'],
#             "class_confusion": class_info['class_confusion_matrices'],
#             "task_accy": task_info['all_accy'],
#             "task_confusion": task_info['task_confusion_matrices'],
#             "all_task_accy": task_info['task_accy'],
#         }







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
from utils.toolkit import tensor2numpy
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

EPSILON = 1e-8

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        # Worker = 0 hoặc thấp để an toàn RAM
        self.num_workers = min(args["num_workers"], 2) 

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
        gc.collect() # Quan trọng: Dọn rác RAM
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
        self._clear_gpu()

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
        
        # Train Noise/GPM
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # --- RLS Training with Pseudo-Replay ---
        # Dùng batch nhỏ (512) để an toàn RAM
        safe_rls_batch = 512
        
        # Fit lần 1 (Augmented) - Có thể bỏ nếu muốn nhanh
        rls_loader = DataLoader(train_set, batch_size=safe_rls_batch, shuffle=True,
                                  num_workers=self.num_workers)
        self.fit_fc(rls_loader, test_loader)

        # Re-fit (No Aug) - Đây là bước quan trọng nhất
        train_set_noaug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_noaug.labels = self.cat2order(train_set_noaug.labels, data_manger)
        
        rls_loader_noaug = DataLoader(train_set_noaug, batch_size=safe_rls_batch, shuffle=True,
                                        num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(rls_loader_noaug, test_loader)
        
        del train_set, test_set, train_set_noaug
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

        # -----------------------------------------------------------
        # [KHÔI PHỤC LẠI LOGIC GỐC]
        # 1. Update Normal FC
        self._network.update_fc(self.increment)
        
        # 2. GỌI FIT NGAY TẠI ĐÂY (Để mở rộng RLS Weight tự động)
        # Hàm fit() trong inc_net.py sẽ tự phát hiện class mới và cat thêm cột
        # Đồng thời nó cũng sinh mẫu giả sơ bộ -> Rất tốt để làm mốc
        self.fit_fc(train_loader, test_loader)
        # -----------------------------------------------------------

        # 3. Train Noise (Giờ chạy Run thoải mái vì Weight đã được expand ở bước 2)
        train_loader_noise = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        self._network.update_noise()
        
        self._clear_gpu()
        self.run(train_loader_noise) # ---> HẾT LỖI
        
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()

        del train_set

        # 4. RE-FIT (TINH CHỈNH CUỐI CÙNG)
        # Sau khi train Noise xong, Feature bị trôi đi một chút.
        # Ta gọi fit lại lần nữa để sinh mẫu giả khớp với Feature mới nhất.
        train_set_noaug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_noaug.labels = self.cat2order(train_set_noaug.labels, data_manger)

        # Batch 512 cho nhanh
        rls_loader = DataLoader(train_set_noaug, batch_size=512, shuffle=True,
                                num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters():
                param.requires_grad = False

        self.re_fit(rls_loader, test_loader)
        
        del train_set_noaug, test_set
        self._clear_gpu()
    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)

        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            # Không cần to(device) vì RLS giờ chạy trên CPU
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
                self._network.fit(inputs, targets)
            
            info = "Task {} --> Update Analytical Classifier!".format(self.cur_task)
            self.logger.info(info)
            prog_bar.set_description(info)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit(inputs, targets)

            info = "Task {} --> Reupdate Analytical Classifier!".format(self.cur_task)
            prog_bar.set_description(info)
        
        self._clear_gpu()

    def run(self, train_loader):
        # ... (Giữ nguyên phần train noise/GPM của bạn) ...
        # (Chỉ copy lại đoạn cũ để đảm bảo không lỗi cú pháp)
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay
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
        WARMUP_EPOCHS = 5

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
            info = "Task {} | Ep {}/{} | Loss {:.3f} | Acc {:.2f}".format(self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)
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