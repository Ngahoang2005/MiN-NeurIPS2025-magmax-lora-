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
from utils.training_tool import get_optimizer, get_scheduler
from torch.amp import autocast, GradScaler

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
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        print('total acc: {}'.format(self.total_acc))
        
        if self.cur_task >= 0:
            self.analyze_cosine_accuracy(test_loader)
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
        train_list, test_list, _ = data_manger.get_task_list(0)
        self.logger.info(f"Task 0 Order: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        # 1. Calculate Proto (MODE = CUR_TASK)
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        
        # 2. Train Noise (MODE = CUR_TASK)
        self.run(train_loader)
        
        # Update Proto (MODE = CUR_TASK)
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        
        # 3. Fit Classifier (MODE = CUR_TASK)
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(train_loader_buf, test_loader) 

        # 4. Re-Fit (MODE = CUR_TASK)
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        
        self.re_fit(train_loader_clean, test_loader)
        del train_set, test_set, train_set_clean; self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)
        self.logger.info(f"Task {self.cur_task}: {train_list}")

        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        
        self._network.update_fc(self.increment)
        self._network.update_noise()

        # Fit Warm-up (MODE = CUR_TASK)
        self.fit_fc(train_loader, test_loader)
        
        # Train Expert (MODE = CUR_TASK)
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader_run)
        self._network.extend_task_prototype(prototype)
        
        self.run(train_loader_run)
        
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader_run)
        self._network.update_task_prototype(prototype)

        del train_set

        # Re-Fit (MODE = CUR_TASK)
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
            
        self.re_fit(train_loader_clean, test_loader) 

        del train_set_clean, test_set; self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # [FIX] Luôn set mode Expert khi fit để học feature đã biến đổi
        self._network.set_noise_mode(self.cur_task) 
        
        prog_bar = tqdm(range(self.fit_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.to(self.device)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
                self._network.fit(inputs, targets)
            
            info = f"Task {self.cur_task} --> Fast RLS Update"
            self.logger.info(info); prog_bar.set_description(info)
            self._clear_gpu()

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        
        # [FIX] Luôn set mode Expert khi refit
        self._network.set_noise_mode(self.cur_task)

        prog_bar = tqdm(train_loader, desc="Refit RLS")
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit(inputs, targets)

        self.logger.info(f"Task {self.cur_task} --> Classifier Refined!")
        self._clear_gpu()

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

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
        self._network.set_noise_mode(self.cur_task) 
        
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct = 0; total = 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 

                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits2 = logits2 + logits1
                        loss_ce = F.cross_entropy(logits2, targets.long())
                        logits_final = logits2
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs["logits"]
                        loss_ce = F.cross_entropy(logits, targets.long())
                        logits_final = logits

                    # ORTHO LOSS
                    loss_orth = torch.tensor(0.0, device=self.device)
                    if self.cur_task > 0:
                        for m in self._network.backbone.noise_maker:
                            curr_mu = m.mu[self.cur_task].weight.flatten()
                            prev_mus = []
                            for t in range(self.cur_task):
                                prev_mus.append(m.mu[t].weight.flatten())
                            if len(prev_mus) > 0:
                                prev_stack = torch.stack(prev_mus).detach() 
                                curr_norm = F.normalize(curr_mu.unsqueeze(0), p=2, dim=1)
                                prev_norm = F.normalize(prev_stack, p=2, dim=1)
                                cos_sim = torch.mm(curr_norm, prev_norm.t())
                                loss_orth += torch.sum(torch.abs(cos_sim))

                    lambda_orth = 0.1 
                    loss = loss_ce + lambda_orth * loss_orth

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets

            scheduler.step()
            train_acc = 100. * correct / total
            info = f"Task {self.cur_task} Ep {epoch+1}: Loss {losses/len(train_loader):.3f} (Orth {loss_orth.item():.4f}) Acc {train_acc:.2f}%"
            self.logger.info(info); prog_bar.set_description(info)
            if epoch % 5 == 0: self._clear_gpu()

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                # Inference dùng Combined (Universal + Expert)
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
        
        # [FIX] Lấy prototype thông qua con mắt của Expert
        model.set_noise_mode(self.cur_task) 
        
        all_features = []
        all_targets = []
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                all_features.append(feature.detach().cpu())
                all_targets.append(targets.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        unique_classes = torch.unique(all_targets).sort()[0]
        class_prototypes = []
        for c in unique_classes:
            mask = (all_targets == c)
            class_mean = all_features[mask].mean(dim=0)
            class_prototypes.append(class_mean)
            
        prototypes = torch.stack(class_prototypes).to(self.device)
        self._clear_gpu()
        return prototypes

    def analyze_cosine_accuracy(self, test_loader):
        self._network.eval()
        all_sims = []
        all_corrects = []
        
        curr_protos = self._network.task_prototypes[self.cur_task].to(self.device)
        curr_protos_norm = F.normalize(curr_protos, p=2, dim=1)

        print(f">>> [DEBUG] Analyzing Class-based Cosine vs Acc for Task {self.cur_task}...")
        with torch.no_grad():
            for _, inputs, targets in tqdm(test_loader, desc="Testing Bins"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # So sánh: Feature qua Expert vs Prototype (cũng từ Expert)
                self._network.set_noise_mode(self.cur_task) 
                feat = self._network.extract_feature(inputs)
                feat_norm = F.normalize(feat, p=2, dim=1)
                
                sim_matrix = torch.mm(feat_norm, curr_protos_norm.t())
                sim, _ = sim_matrix.max(dim=1)
                
                outputs = self._network.forward_tuna_combined(inputs)
                preds = outputs['logits'].argmax(dim=1)
                correct = (preds == targets).float()

                all_sims.extend(sim.cpu().numpy())
                all_corrects.extend(correct.cpu().numpy())

        self._plot_cosine_acc_chart(all_sims, all_corrects)

    def _plot_cosine_acc_chart(self, sims, corrects, num_bins=10):
        import matplotlib.pyplot as plt
        sims, corrects = np.array(sims), np.array(corrects)
        bin_edges = np.linspace(0, 1, num_bins + 1)
        acc_per_bin = []
        bin_centers = []
        for i in range(num_bins):
            mask = (sims >= bin_edges[i]) & (sims < bin_edges[i+1])
            if mask.any():
                acc_per_bin.append(corrects[mask].mean() * 100)
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, acc_per_bin, width=0.08, color='skyblue', edgecolor='black', alpha=0.7)
        plt.plot(bin_centers, acc_per_bin, marker='o', color='blue', linewidth=2)
        plt.xlabel('Cosine Similarity (Expert Space)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Task {self.cur_task}: Expert-based Sim vs Acc', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.ylim(0, 105)
        plt.savefig(f'cosine_acc_task_{self.cur_task}.png')
        plt.close()