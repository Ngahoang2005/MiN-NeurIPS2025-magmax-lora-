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
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = 0 # Ép về 0 để chống treo máy

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

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def calculate_drift(self, clean_loader):
        self._network.eval(); self._network.to(self.device)
        self._old_network.eval(); self._old_network.to(self.device)
        bs = self.buffer_size
        XtX = torch.zeros(bs, bs, device='cpu')
        XtY = torch.zeros(bs, bs, device='cpu')
        
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                # [FIX TYPE]: To float()
                f_old = self._old_network.buffer(self._old_network.backbone(inputs.to(self.device)).float()).cpu()
                f_new = self._network.buffer(self._network.backbone(inputs.to(self.device)).float()).cpu()
                XtX += f_old.t() @ f_old
                XtY += f_old.t() @ f_new
        
        reg = 1e-4 * torch.eye(bs)
        P = torch.linalg.solve(XtX + reg, XtY)
        return P.to(self.device)

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=0)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        # 1. Run training
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # 2. Fit FC (Giai đoạn 1)
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0)
        self.fit_fc(fit_loader, test_loader=None) # Giữ đúng hàm của bạn

        # 3. Re-fit với no_aug
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=0)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_no_aug, None)
     

    def increment_train(self, data_manger):
        self.cur_task += 1
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=0)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        # LOGIC CỦA BẠN: Fit trước
        self.fit_fc(train_loader, None)

        # Update model
        self._network.update_fc(self.increment)
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._network.update_noise()
        self._clear_gpu()

        # Run training
        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()

        # LOGIC CỦA BẠN: Re-fit sau với no_aug
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=0)

        self.re_fit(train_loader_no_aug, None)
       

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        # [STREAMING FIT]: Chia nhỏ theo class để chống OOM 16k
        dataset = train_loader.dataset
        classes = sorted(list(set(dataset.labels)))
        
        for c in classes:
            indices = np.where(np.array(dataset.labels) == c)[0]
            sub_loader = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch)
            for _ in range(self.fit_epoch):
                for _, inputs, targets in sub_loader:
                    y_onehot = F.one_hot(targets.to(self.device), num_classes=self._network.known_class)
                    self._network.accumulate_stats(inputs.to(self.device), y_onehot)
            # Nén từng class một
            self._network.compress_stats()
            self._clear_gpu()
        
        self._network.fit(init_mode=True if self.cur_task==0 else False)

    def re_fit(self, train_loader, test_loader):
        P_drift = None
        if self.cur_task > 0:
            P_drift = self.calculate_drift(train_loader)
        
        boundary = self.known_class - self.increment if self.cur_task > 0 else 0
        self._network.fit(P_drift=P_drift, known_classes_boundary=boundary, init_mode=False)
        self._clear_gpu()

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, self.args['weight_decay'])
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        self._network.train()
        for epoch in range(epochs):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    l2 = self._network.forward_normal_fc(inputs)['logits']
                    if self.cur_task > 0:
                        with torch.no_grad(): l1 = self._network(inputs)['logits']
                        # [FIX TYPE]: l1.to(l2.dtype)
                        if l2.shape[1] > l1.shape[1]:
                            padding = torch.zeros((l1.shape[0], l2.shape[1]-l1.shape[1]), device=self.device, dtype=l1.dtype)
                            l1 = torch.cat([l1, padding], dim=1)
                        logits = l2 + l1.to(l2.dtype)
                    else:
                        logits = l2
                    loss = F.cross_entropy(logits, targets.long())

                self.scaler.scale(loss).backward()
                if self.cur_task > 0: 
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=0.85)
                self.scaler.step(optimizer)
                self.scaler.update()
            scheduler.step()
        self._clear_gpu()

    @staticmethod
    def cat2order(targets, datamanger): return [datamanger.map_cat2order(t) for t in targets]

    def eval_task(self, test_loader):
        self._network.eval(); pred, label = [], []
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                logits = self._network(inputs.to(self.device))["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend(predicts.cpu().numpy())
                label.extend(targets.numpy())
        
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
        }

    def after_train(self, data_manger):
        if self.cur_task == 0: self.known_class = self.init_class
        else: self.known_class += self.increment
        
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        eval_res = self.eval_task(DataLoader(test_set, batch_size=self.init_batch_size, num_workers=0))
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('Avg acc: {:.2f}'.format(sum(self.total_acc)/len(self.total_acc)))
        print('Total Acc: {}'.format(self.total_acc))
        print('Avg Acc: {:.2f}'.format(sum(self.total_acc)/len(self.total_acc)))
        self._clear_gpu()