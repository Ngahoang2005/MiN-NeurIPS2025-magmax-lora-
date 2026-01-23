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
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
# Mixed Precision (giữ lại vì hữu ích)
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
        self.fit_epoch = args["fit_epochs"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']  # giữ nếu có trong args

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        
        self.scaler = GradScaler('cuda')

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

        self._network.update_fc(self.init_class)
        self._network.update_noise()  # Kích hoạt BiLoRA

        self._network.to(self.device)
        self._clear_gpu()

        self.run(train_loader)  # Train noise (BiLoRA)

        self._network.after_task_magmax_merge()  # Merge BiLoRA

        self._clear_gpu()

        # Fit analytic trên augmented data
        train_loader_fit = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True,
                                      num_workers=self.num_workers)
        self.fit_fc(train_loader_fit)

        # Re-fit trên clean data (no aug)
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)
        self.re_fit(train_loader_no_aug)

        del train_set, train_set_no_aug, test_set
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

        # Warmup analytic trước khi train noise mới
        self.fit_fc(train_loader)

        self._network.update_fc(self.increment)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)

        self._network.update_noise()  # Kích hoạt BiLoRA mới
        self._network.to(self.device)
        self._clear_gpu()

        self.run(train_loader)  # Train noise

        self._network.after_task_magmax_merge()  # Merge

        self._clear_gpu()

        # Re-fit trên clean data
        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True,
                                         num_workers=self.num_workers)
        self.re_fit(train_loader_no_aug)

        del train_set, train_set_no_aug, test_set
        self._clear_gpu()

    def fit_fc(self, train_loader):
        self._network.eval()
        prog_bar = tqdm(range(self.fit_epoch))
        for epoch in prog_bar:
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = F.one_hot(targets, num_classes=self._network.normal_fc.out_features).float()
                self._network.fit(inputs, targets)
            self._clear_gpu()

    def re_fit(self, train_loader):
        self._network.eval()
        self._network.to(self.device)
        prog_bar = tqdm(train_loader)
        for _, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = F.one_hot(targets, num_classes=self._network.normal_fc.out_features).float()
            self._network.fit(inputs, targets)
        self._clear_gpu()

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = 5  # sát gốc
            lr = self.lr * 0.1  # sát gốc
            weight_decay = self.weight_decay

        # Freeze all trừ normal_fc và noise (BiLoRA)
        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True

        if self.cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()

        params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        self._network.train()
        self._network.to(self.device)

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits = logits1 + logits2
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs["logits"]

                    loss = F.cross_entropy(logits, targets.long())

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += len(targets)

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}".format(
                self.cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            self.logger.info(info)

            if epoch % 5 == 0:
                self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend(predicts.cpu().numpy().tolist())
                label.extend(targets.cpu().numpy().tolist())
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices']
        }