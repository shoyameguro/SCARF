import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import CorruptionDataset, get_dataset
from infonce import InfoNCE
from model import Classification, SCARFSelf, UnlabeledLoss
from utils import (EarlyStopping, corruption_generator, negative_sample,
                   perf_metric, pretext_generator)

log = logging.getLogger(__name__)


class Train:
    def __init__(self, config) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.l_set, self.u_set, self.test_set = get_dataset(config['data_name'], config['label_data_rate'])
        self.self_epochs = config['self_epochs']
        self.semi_max_iter = config['semi_max_iter']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.k = config['k']
        self.c = config['c']
        self.temperature = config['temperature']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.l_samples = len(self.l_set)
        self.dim = self.l_set[0][0].shape[-1]
        self.l_dim = self.l_set[0][1].shape[-1]

        self.scarf_self = SCARFSelf(self.dim, self.dim).to(self.device)
        self.classification = Classification(self.dim, self.l_dim).to(self.device)

        self.l_loss_fn = nn.CrossEntropyLoss()

        self.u_loss_fn = UnlabeledLoss()
        self.info_nce = InfoNCE(self.temperature)

        self.opt_self = optim.SGD(self.scarf_self.parameters(), lr=1e-3)
        self.opt_semi = optim.Adam(self.classification.parameters())

        self.scheduler = StepLR(self.opt_self, step_size=50, gamma=0.1)
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.opt_self, lambda epoch: (epoch+1)/10.0, verbose=False)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_self, 500, eta_min=0.05, last_epoch=-1, verbose=False)
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    def self_ul(self):
        x = self.u_set.x.detach()
        corruption = corruption_generator(x.shape, self.c)
        corruption = corruption.to(x.device)
        corruption, x_tilde = pretext_generator(corruption, x)

        u_loader = DataLoader(CorruptionDataset(x, x_tilde, corruption), 512, shuffle=True)
        for e in range(self.self_epochs):
            with tqdm(u_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
                for positive_idx, data in enumerate(pbar_epoch):
                    x, x_tilde, corruption = data
                    x = x.to(self.device)
                    x_tilde = x_tilde.to(self.device)
                    corruption = corruption.to(self.device)
                    self.opt_self.zero_grad()

                    z_i, z_j, reconstruction_loss = self.scarf_self(x, x_tilde)
                    negative_samples = negative_sample(x, positive_idx)
                    negative_keys = torch.zeros_like(negative_samples)

                    for num, x in enumerate(negative_samples):
                        x = x.to(self.device)
                        x = self.scarf_self.encoder(x)
                        negative_keys[num] = x

                    # loss = self.info_nce(z_i, z_j, negative_keys)
                    contrastive_loss = self.info_nce(z_i, z_j)
                    loss = contrastive_loss + self.alpha * reconstruction_loss
                    loss.backward()
                    self.opt_self.step()
                    pbar_epoch.set_description(f"epoch[{e + 1} / {self.self_epochs}]")
                    pbar_epoch.set_postfix({'loss': loss.item(),
                                            'contrastiveloss': contrastive_loss.item(),
                                            'reconstructionloss': reconstruction_loss.item()})
            if e < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

    def sl(self):
        idx = np.random.permutation(len(self.l_set))
        train_idx = idx[:int(len(idx) * 0.9)]
        valid_idx = idx[int(len(idx) * 0.9):]

        val_set = self.l_set[valid_idx]

        for i in range(self.semi_max_iter):
            b_idx = np.random.permutation(len(train_idx))[:self.batch_size].tolist()
            l_batch = self.l_set[train_idx[b_idx]]
            x_batch, y_batch = l_batch
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                x_batch = self.scarf_self.encoder(x_batch)
            self.opt_semi.zero_grad()
            l_pred = self.classification(x_batch)
            loss = self.l_loss_fn(l_pred, y_batch)
            loss.backward()
            self.opt_semi.step()

            with torch.no_grad():
                x, y = val_set
                x = x.to(self.device)
                y = y.to(self.device)
                x = self.scarf_self.encoder(x)
                pred = self.classification(x)
                val_loss = self.l_loss_fn(pred, y)

            if i % 100 == 0:
                print(f'Iteration: {i} / {self.semi_max_iter}, '
                      f'Current loss (val): {val_loss.item(): .4f}, '
                      f'Current loss (train): {loss.item(): .4f}')

            if i % math.ceil(self.l_samples / self.batch_size) == 0:
                self.early_stopping(val_loss, self.classification)
                if self.early_stopping.early_stop:
                    print(f'early stopping {i} / {self.semi_max_iter}')
                    self.classification.load_state_dict(torch.load('checkpoint.pt'))
                    break

    def semi_sl(self):
        idx = np.random.permutation(len(self.l_set))
        train_idx = idx[:int(len(idx) * 0.9)]
        valid_idx = idx[int(len(idx) * 0.9):]

        val_set = self.l_set[valid_idx]

        for i in range(self.semi_max_iter):
            b_idx = np.random.permutation(len(train_idx))[:self.batch_size].tolist()
            l_batch = self.l_set[train_idx[b_idx]]
            x_batch, y_batch = l_batch
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                x_batch = self.scarf_self.encoder(x_batch)

            bu_idx = np.random.permutation(len(self.u_set))[:self.batch_size]
            u_batch = self.u_set[bu_idx]
            xu_batch_ori, _ = u_batch
            xu_batch_ori = xu_batch_ori.to(self.device)

            xu_batch = []
            for _ in range(self.k):
                m_batch = corruption_generator(xu_batch_ori.shape, self.c)
                m_batch = m_batch.to(self.device)
                _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

                with torch.no_grad():
                    xu_batch_temp = self.scarf_self.encoder(xu_batch_temp)
                xu_batch.append(xu_batch_temp.unsqueeze(0))
            xu_batch = torch.cat(xu_batch)

            self.opt_semi.zero_grad()
            l_pred = self.classification(x_batch)
            u_pred = self.classification(xu_batch)

            l_loss = self.l_loss_fn(l_pred, y_batch)
            u_loss = self.u_loss_fn(u_pred)
            loss = l_loss + self.beta * u_loss
            loss.backward()
            self.opt_semi.step()

            with torch.no_grad():
                x, y = val_set
                x = x.to(self.device)
                y = y.to(self.device)
                x = self.scarf_self.encoder(x)
                pred = self.classification(x)
                val_loss = self.l_loss_fn(pred, y)

            if i % 100 == 0:
                print(f'Iteration: {i} / {self.semi_max_iter}, '
                      f'Current loss (val): {val_loss.item(): .4f}, '
                      f'Current loss (train): {loss.item(): .4f}, '
                      f'supervised loss: {l_loss.item(): .4f}, '
                      f'unsupervised loss: {u_loss.item(): .4f}')

            if i % math.ceil(self.l_samples / self.batch_size) == 0:
                self.early_stopping(val_loss, self.classification)
                if self.early_stopping.early_stop:
                    print(f'early stopping {i} / {self.semi_max_iter}')
                    self.classification.load_state_dict(torch.load('checkpoint.pt'))
                    break

    def test(self):
        test_loader = DataLoader(self.test_set, self.test_batch_size)
        results = []
        with tqdm(test_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
            for data in pbar_epoch:
                with torch.no_grad():
                    x, y = data
                    x = x.to(self.device)

                    x = self.scarf_self.encoder(x)
                    pred = self.classification(x)
                    results.append(perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy()))
        log.info(f'Performance: {100 * torch.tensor(results).mean()}')
