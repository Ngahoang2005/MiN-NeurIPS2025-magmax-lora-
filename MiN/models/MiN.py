import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import copy, gc
from utils.inc_net import MiNbaseNet
from utils.toolkit import calculate_class_metrics
from utils.training_tool import get_optimizer, get_scheduler
from torch.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        self.args, self.logger = args, loger
        self._network = MiNbaseNet(args)
        self.device, self.num_workers = args['device'], 0
        self.init_epochs, self.init_lr = args["init_epochs"], args["init_lr"]
        self.lr, self.batch_size = args["lr"], args["batch_size"]
        self.init_class, self.increment = args["init_class"], args["increment"]
        self.buffer_batch, self.fit_epoch = args["buffer_batch"], args["fit_epochs"]
        self.cur_task, self.total_acc = -1, []
        self.scaler = GradScaler('cuda')
        self._old_network = None 

    def _clear_gpu(self):
        gc.collect()
        torch.cuda.empty_cache()

    def calculate_drift(self, clean_loader):
        self._network.eval(); self._old_network.eval()
        bs = self.args['buffer_size']
        XtX = torch.zeros(bs, bs); XtY = torch.zeros(bs, bs)
        with torch.no_grad():
            for _, inputs, _ in clean_loader:
                f_old = self._old_network.buffer(self._old_network.backbone(inputs.to(self.device)).float()).cpu()
                f_new = self._network.buffer(self._network.backbone(inputs.to(self.device)).float()).cpu()
                XtX += f_old.t() @ f_old; XtY += f_old.t() @ f_new
        return torch.linalg.solve(XtX + 1e-4*torch.eye(bs), XtY).to(self.device)

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        if self.args['pretrained']:
            for p in self._network.backbone.parameters(): p.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self.run(DataLoader(train_set, batch_size=self.args['init_batch_size'], shuffle=True, num_workers=0))
        self._network.collect_projections(val=0.95)
        
        # Fit FC logic
        self.fit_fc(DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0))
        
        # No Aug Refit
        train_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_no_aug.labels = self.cat2order(train_no_aug.labels, data_manger)
        if self.args['pretrained']:
            for p in self._network.backbone.parameters(): p.requires_grad = False
        self.re_fit(DataLoader(train_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=0))
        

    def increment_train(self, data_manger):
        self.cur_task += 1
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        train_list, _, _ = data_manger.get_task_list(self.cur_task)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)

        # LOGIC GỐC: Fit trước, Update FC, Run, Re-fit sau
        self.fit_fc(DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=0), init_mode=True)
        self._network.update_fc(self.increment)
        self._network.update_noise()
        self.run(DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0))
        self._network.collect_projections(val=0.95)

        train_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_no_aug.labels = self.cat2order(train_no_aug.labels, data_manger)
        self.re_fit(DataLoader(train_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=0))


    def fit_fc(self, loader, init_mode=False):
        self._network.eval()
        if init_mode:
            for _ in range(self.fit_epoch):
                for _, x, y in loader: self._network.accumulate_stats(x.to(self.device), F.one_hot(y.to(self.device), self._network.known_class))
            self._network.fit(init_mode=True)
        else:
            dataset = loader.dataset
            classes = sorted(list(set(dataset.labels)))
            for c in classes:
                indices = np.where(np.array(dataset.labels) == c)[0]
                sub = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch)
                for _ in range(self.fit_epoch):
                    for _, x, y in sub: self._network.accumulate_stats(x.to(self.device), F.one_hot(y.to(self.device), self._network.known_class))
                self._network.compress_stats(); self._clear_gpu()
            self._network.fit(init_mode=False)

    def re_fit(self, loader):
        P = self.calculate_drift(loader) if self.cur_task > 0 else None
        self._network.fit(P_drift=P, known_classes_boundary=(self._network.known_class - self.increment), init_mode=False)
        self._clear_gpu()

    def run(self, loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        opt = get_optimizer(self.args['optimizer_type'], params, (self.init_lr if self.cur_task==0 else self.lr), self.args['weight_decay'])
        sch = get_scheduler(self.args['scheduler_type'], opt, epochs)
        self._network.train()
        for epoch in range(epochs):
            for _, x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                with autocast('cuda'):
                    l2 = self._network.forward_normal_fc(x)['logits']
                    if self.cur_task > 0:
                        with torch.no_grad(): l1 = self._network(x)['logits']
                        if l2.shape[1] > l1.shape[1]: l1 = torch.cat([l1, torch.zeros((l1.shape[0], l2.shape[1]-l1.shape[1]), device=self.device)], dim=1)
                        logits = l2 + l1.to(l2.dtype)
                    else: logits = l2
                    loss = F.cross_entropy(logits, y.long())
                self.scaler.scale(loss).backward()
                if self.cur_task > 0: 
                    self.scaler.unscale_(opt); self._network.apply_gpm_to_grads(scale=0.85)
                self.scaler.step(opt); self.scaler.update()
            sch.step()
        self._clear_gpu()

    @staticmethod
    def cat2order(targets, datamanger): return [datamanger.map_cat2order(t) for t in targets]

    def after_train(self, data_manger):
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        res = self.eval_task(DataLoader(test_set, batch_size=self.args['init_batch_size'], num_workers=0))
        self.total_acc.append(round(res['all_accy']*100, 2))
        self.logger.info(f'total acc: {self.total_acc}'); self._clear_gpu()

    def eval_task(self, loader):
        self._network.eval(); pred, label = [], []
        with torch.no_grad():
            for _, x, y in loader:
                logits = self._network(x.to(self.device))["logits"]
                pred.extend(torch.max(logits, 1)[1].cpu().numpy()); label.extend(y.numpy())
        return calculate_class_metrics(pred, label)