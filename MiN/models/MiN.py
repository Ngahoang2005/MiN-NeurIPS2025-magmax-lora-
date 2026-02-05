import math, random, gc, os
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args, self.logger = args, loger
        self.device = args['device']
        self._network = MiNbaseNet(args).to(self.device)
        self.num_workers = args["num_workers"]
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
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info(f"task_list: {train_list_name}")
        self.logger.info(f"task_order: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        
        self.run(train_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()
        
        # [GIỮ NGUYÊN]: Fit FC bằng train_set
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, self.test_loader)

        # [GIỮ NGUYÊN]: Re-fit bằng train_no_aug
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.re_fit(train_loader_no_aug, self.test_loader)
        del train_set, test_set
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"task_list: {train_list_name}")
        self.logger.info(f"task_order: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        # [GIỮ NGUYÊN]: Fit trước
        self.fit_fc(train_loader, self.test_loader)
        self._network.update_fc(self.increment)
        
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()

        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()

        del train_set
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)

        # [GIỮ NGUYÊN]: Re-fit sau
        self.re_fit(train_loader_no_aug, self.test_loader)
        del train_set_no_aug, test_set
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        """Fit FC: Logic DPCR thay thế bên trong vòng lặp của mày"""
        self._network.eval()
        dataset = train_loader.dataset
        classes = sorted(list(set(dataset.labels)))
        # [DPCR]: Gom và nén theo từng class để tiết kiệm VRAM
        for c in classes:
            idx = np.where(np.array(dataset.labels) == c)[0]
            sub = DataLoader(Subset(dataset, idx), batch_size=self.buffer_batch)
            for _ in range(self.fit_epoch):
                for _, x, y in sub:
                    y_oh = F.one_hot(y.to(self.device), self._network.known_class)
                    self._network.fit(x, y_oh)
            self._network.compress_stats(); self._clear_gpu()
        self._network.solve_analytic(init_mode=(self.cur_task==0))

    def re_fit(self, train_loader, test_loader):
        """Re-fit: Có Drift Correction theo chuẩn DPCR"""
        P = self.calculate_drift(train_loader) if self.cur_task > 0 else None
        boundary = self.known_class - self.increment if self.cur_task > 0 else 0
        for _, x, y in train_loader:
            y_oh = F.one_hot(y.to(self.device), self._network.known_class)
            self._network.fit(x, y_oh)
        self._network.solve_analytic(P_drift=P, boundary=boundary, init_mode=False)
        self._clear_gpu()

    def run(self, train_loader):
        # [GIỮ NGUYÊN]: Toàn bộ logic train Gradient Descent của mày
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        optimizer = get_optimizer(self.args['optimizer_type'], filter(lambda p: p.requires_grad, self._network.parameters()), 
                                  (self.init_lr if self.cur_task==0 else self.lr), self.args['weight_decay'])
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
                        if l2.shape[1] > l1.shape[1]:
                            pad = torch.zeros((l1.shape[0], l2.shape[1]-l1.shape[1]), device=self.device, dtype=l1.dtype)
                            l1 = torch.cat([l1, pad], dim=1)
                        logits = l2 + l1.to(l2.dtype)
                    else: logits = l2
                    loss = F.cross_entropy(logits, targets.long())
                self.scaler.scale(loss).backward()
                if self.cur_task > 0: 
                    self.scaler.unscale_(optimizer); self._network.apply_gpm_to_grads(scale=0.85)
                self.scaler.step(optimizer); self.scaler.update()
            scheduler.step()
        self._clear_gpu()

    def after_train(self, data_manger):
        # [GIỮ NGUYÊN]: In kết quả acc/avg_acc theo mẫu mày gửi
        if self.cur_task == 0: self.known_class = self.init_class
        else: self.known_class += self.increment
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        eval_res = self.eval_task(DataLoader(test_set, batch_size=self.init_batch_size, num_workers=0))
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        
        self.logger.info(f'total acc: {self.total_acc}')
        self.logger.info(f'avg_acc: {np.mean(self.total_acc):.2f}')
        print(f'total acc: {self.total_acc}')
        print(f'avg_acc: {np.mean(self.total_acc):.2f}')
        self._clear_gpu()

    def eval_task(self, test_loader):
        self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                logits = self._network(inputs.to(self.device))["logits"]
                pred.extend(torch.max(logits, 1)[1].cpu().numpy()); label.extend(targets.numpy())
        res = calculate_class_metrics(pred, label)
        task_res = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {"all_class_accy": res['all_accy'], "task_confusion": task_res['task_confusion_matrices']}

    @staticmethod
    def cat2order(targets, datamanger): return [datamanger.map_cat2order(t) for t in targets]