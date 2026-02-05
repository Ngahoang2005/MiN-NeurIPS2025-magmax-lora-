import math, random, gc, os
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from utils.inc_net import MiNbaseNet
from utils.toolkit import calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args, self.logger, self.device = args, loger, args['device']
        self._network = MiNbaseNet(args).to(self.device)
        self.num_workers = args["num_workers"] # Set về 0 trong config nếu vẫn treo
        self.init_epochs, self.init_lr = args["init_epochs"], args["init_lr"]
        self.lr, self.batch_size = args["lr"], args["batch_size"]
        self.init_class, self.increment = args["init_class"], args["increment"]
        self.buffer_size, self.buffer_batch = args["buffer_size"], args["buffer_batch"]
        self.fit_epoch = args["fit_epochs"]
        self.known_class, self.cur_task = 0, -1
        self.total_acc = []
        self.scaler = GradScaler('cuda')
        self._old_network = None 

    def _clear_gpu(self):
        gc.collect(); torch.cuda.empty_cache()

    def calculate_drift(self, clean_loader):
        self._network.eval(); self._old_network.eval()
        bs = self.buffer_size
        XtX = torch.zeros(bs, bs, device=self.device)
        XtY = torch.zeros(bs, bs, device=self.device)
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                inputs = inputs.to(self.device)
                f_old = self._old_network.buffer(self._old_network.backbone(inputs.float()).float())
                f_new = self._network.buffer(self._network.backbone(inputs.float()).float())
                XtX += f_old.t() @ f_old; XtY += f_old.t() @ f_new
        return torch.linalg.solve(XtX + 1e-4*torch.eye(bs, device=self.device), XtY)

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, _, _ = data_manger.get_task_list(0)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True
        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        self.run(DataLoader(train_set, batch_size=self.args['init_batch_size'], shuffle=True, num_workers=self.num_workers))
        self._network.collect_projections(val=0.95)
        self.fit_fc(DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers), None)
        
        train_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_no_aug.labels = self.cat2order(train_no_aug.labels, data_manger)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        self.re_fit(DataLoader(train_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers), None)

    def increment_train(self, data_manger):
        self.cur_task += 1
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        train_list, _, _ = data_manger.get_task_list(self.cur_task)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.fit_fc(DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers), None)
        self._network.update_fc(self.increment)
        self._network.update_noise()
        self.run(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))
        self._network.collect_projections(val=0.95)
        
        train_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_no_aug.labels = self.cat2order(train_no_aug.labels, data_manger)
        self.re_fit(DataLoader(train_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers), None)

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        dataset = train_loader.dataset
        classes = sorted(list(set(dataset.labels)))
        for c in classes:
            idx = np.where(np.array(dataset.labels) == c)[0]
            sub = DataLoader(Subset(dataset, idx), batch_size=self.buffer_batch)
            for _ in range(self.fit_epoch):
                for _, x, y in sub:
                    self._network.accumulate_stats(x, F.one_hot(y.to(self.device), self._network.known_class))
            self._network.compress_stats(); self._clear_gpu()
        self._network.solve_dpcr(init_mode=(self.cur_task==0))

    def re_fit(self, train_loader, test_loader):
        P = self.calculate_drift(train_loader) if self.cur_task > 0 else None
        boundary = self.known_class - self.increment if self.cur_task > 0 else 0
        for _, x, y in train_loader:
            self._network.accumulate_stats(x, F.one_hot(y.to(self.device), self._network.known_class))
        self._network.solve_dpcr(P_drift=P, boundary=boundary, init_mode=False)
        self._clear_gpu()

    def run(self, train_loader):
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        opt = get_optimizer(self.args['optimizer_type'], params, (self.init_lr if self.cur_task==0 else self.lr), self.args['weight_decay'])
        sch = get_scheduler(self.args['scheduler_type'], opt, (self.init_epochs if self.cur_task==0 else self.epochs))
        self._network.train()
        for epoch in range(self.init_epochs if self.cur_task==0 else self.epochs):
            for _, x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                with autocast('cuda'):
                    l2 = self._network.forward_normal_fc(x)['logits']
                    if self.cur_task > 0:
                        with torch.no_grad(): l1 = self._network(x)['logits']
                        if l2.shape[1] > l1.shape[1]:
                            pad = torch.zeros((l1.shape[0], l2.shape[1]-l1.shape[1]), device=self.device, dtype=l1.dtype)
                            l1 = torch.cat([l1, pad], dim=1)
                        logits = l2 + l1.to(l2.dtype)
                    else: logits = l2
                    loss = F.cross_entropy(logits, y.long())
                self.scaler.scale(loss).backward()
                if self.cur_task > 0: 
                    self.scaler.unscale_(opt); self._network.apply_gpm_to_grads(scale=0.85)
                self.scaler.step(opt); self.scaler.update()
            sch.step()
        self._clear_gpu()

    def after_train(self, data_manger):
        if self.cur_task == 0: self.known_class = self.init_class
        else: self.known_class += self.increment
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        res = self.eval_task(DataLoader(test_set, batch_size=self.init_batch_size, num_workers=0))
        self.total_acc.append(round(float(res['all_class_accy']*100.), 2))
        print(f'total acc: {self.total_acc}\navg_acc: {np.mean(self.total_acc):.2f}')
        self.logger.info(f'total acc: {self.total_acc}')
        self.logger.info(f'avg_acc: {np.mean(self.total_acc):.2f}') 
        print('avg_acc:', np.mean(self.total_acc))
    def eval_task(self, loader):
        self._network.eval(); pred, label = [], []
        with torch.no_grad():
            for _, x, y in loader:
                logits = self._network(x.to(self.device))["logits"]
                pred.extend(torch.max(logits, 1)[1].cpu().numpy()); label.extend(y.numpy())
        res = calculate_class_metrics(pred, label)
        task_res = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {"all_class_accy": res['all_accy'], "task_confusion": task_res['task_confusion_matrices']}

    @staticmethod
    def cat2order(targets, datamanger): return [datamanger.map_cat2order(t) for t in targets]