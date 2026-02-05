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

# Mixed Precision
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
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def after_train(self, data_manger):
        if self.cur_task == 0: self.known_class = self.init_class
        else: self.known_class += self.increment

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
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                outputs = model(inputs.to(self.device))
                predicts = torch.max(outputs["logits"], dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)): targets[i] = datamanger.map_cat2order(targets[i])
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
        
        fit_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(fit_loader, self.test_loader)

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
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.fit_fc(train_loader, self.test_loader)
        self._network.update_fc(self.increment)
        
        self._old_network = copy.deepcopy(self._network).to(self.device).eval()
        run_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()

        self.run(run_loader)
        self._network.collect_projections(mode='threshold', val=0.95)
        self._clear_gpu()

        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.re_fit(train_loader_no_aug, self.test_loader)
        del train_set, test_set, train_set_no_aug
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        """Fit FC: Workflow gốc của mày + DPCR nén thống kê"""
        self._network.eval()
        dataset = train_loader.dataset
        classes = sorted(list(set(dataset.labels)))
        prog_bar = tqdm(range(self.fit_epoch))
        for _ in prog_bar:
            for c in classes:
                indices = np.where(np.array(dataset.labels) == c)[0]
                sub = DataLoader(Subset(dataset, indices), batch_size=self.buffer_batch)
                for _, x, y in sub:
                    y_oh = F.one_hot(y.to(self.device), self._network.known_class)
                    self._network.fit(x.to(self.device), y_oh)
                self._network.compress_stats(); self._clear_gpu()
            info = f"Task {self.cur_task} --> Update Analytical Classifier!"
            self.logger.info(info); prog_bar.set_description(info)
        self._network.solve_analytic(init_mode=(self.cur_task==0))

    def re_fit(self, train_loader, test_loader):
        """Re-fit: Workflow gốc của mày + Drift Correction"""
        self._network.eval()
        P = self.calculate_drift(train_loader) if self.cur_task > 0 else None
        boundary = self.known_class - self.increment if self.cur_task > 0 else 0
        prog_bar = tqdm(train_loader)
        for _, x, y in prog_bar:
            y_oh = F.one_hot(y.to(self.device), self._network.known_class)
            self._network.fit(x.to(self.device), y_oh)
            info = f"Task {self.cur_task} --> Reupdate Analytical Classifier!"
            prog_bar.set_description(info)
        self._network.solve_analytic(P_drift=P, boundary=boundary, init_mode=False)
        self._clear_gpu()

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

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, (self.init_lr if self.cur_task==0 else self.lr), self.args['weight_decay'])
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)
        self._network.train()
        current_scale = self.compute_adaptive_scale(train_loader) if self.cur_task > 0 else 0.95
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    l2 = self._network.forward_normal_fc(inputs)['logits']
                    if self.cur_task > 0:
                        with torch.no_grad(): l1 = self._network(inputs)['logits']
                        if l2.shape[1] > l1.shape[1]:
                            l1 = torch.cat([l1, torch.zeros((l1.shape[0], l2.shape[1]-l1.shape[1]), device=self.device, dtype=l1.dtype)], dim=1)
                        logits = l2 + l1.to(l2.dtype)
                    else: logits = l2
                    loss = F.cross_entropy(logits, targets.long())
                self.scaler.scale(loss).backward()
                if self.cur_task > 0 and epoch >= 2:
                    self.scaler.unscale_(optimizer); self._network.apply_gpm_to_grads(scale=0.85)
                self.scaler.step(optimizer); self.scaler.update()
                losses += loss.item()
                correct += torch.max(logits, 1)[1].eq(targets).sum().cpu().item()
                total += len(targets)
            scheduler.step()
            info = f"Task {self.cur_task} | Ep {epoch+1}/{epochs} | Loss {losses/len(train_loader):.3f} | Acc {100.*correct/total:.2f}"
            prog_bar.set_description(info)
        self._clear_gpu()

    def eval_task(self, test_loader):
        self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for _, x, y in test_loader:
                logits = self._network(x.to(self.device))["logits"]
                pred.extend(torch.max(logits, 1)[1].cpu().numpy()); label.extend(y.numpy())
        res = calculate_class_metrics(pred, label)
        task_res = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {"all_class_accy": res['all_accy'], "task_confusion": task_res['task_confusion_matrices']}

    def compute_adaptive_scale(self, loader):
        curr_proto = self.get_task_prototype(self._network, loader)
        if not hasattr(self, 'old_prototypes'): self.old_prototypes = []
        if not self.old_prototypes:
            self.old_prototypes.append(curr_proto); return 0.95
        max_sim = 0.0
        c_norm = F.normalize(curr_proto.unsqueeze(0), p=2, dim=1)
        for op in self.old_prototypes:
            sim = torch.mm(c_norm, F.normalize(op.unsqueeze(0), p=2, dim=1).t()).item()
            if sim > max_sim: max_sim = sim
        self.old_prototypes.append(curr_proto)
        return max(0.65, min(0.5 + 0.5 * (1.0 - max_sim), 0.95))

    def get_task_prototype(self, model, loader):
        model.eval(); feats = []
        with torch.no_grad():
            for _, x, _ in loader:
                with autocast('cuda'): feats.append(model.extract_feature(x.to(self.device)).cpu())
        return torch.mean(torch.cat(feats, 0), 0).to(self.device)