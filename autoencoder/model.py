import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder, self.decoder)

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)

        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class VAE(nn.Module):

    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(VAE, self).__init__()

        self._set_random_seed(42)
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.lr_dev = nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1])
        self.lr_ave = nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1])
        self.relu = nn.ReLU()

        encoder_layers = []
        for i in range(len(encoder_hidden_dims)-2):
            if i == 0:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i+1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i+1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))
                # encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))

            #     # encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
            #     encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
            #     encoder_layers.append(nn.ReLU())
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
            elif i == len(decoder_hidden_dims)-1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
                # decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
                # decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))

        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder)
        print("Various layer: ", self.lr_dev, self.lr_ave)
        print(self.decoder)

    def _set_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    def sampling(self, ave, log_dev):
        std = torch.exp(0.5 * log_dev)
        eps = torch.randn_like(std, memory_format=torch.contiguous_format)
        return eps.mul(std).add_(ave)  # return z sample

    def seed_sampling(self, ave, log_dev, seed=42):
        std = torch.exp(0.5 * log_dev)
        torch.manual_seed(seed)
        eps = torch.randn_like(std, memory_format=torch.contiguous_format)
        return eps.mul(std).add_(ave)  # return z sample

    def forward(self, x):
        for m in self.encoder:
            x = m(x)

        # for VAE
        ave = self.lr_ave(x)  # average
        log_dev = self.lr_dev(x)  # log(sigma^2)

        z = self.sampling(ave, log_dev)
        x = z

        for m in self.decoder:
            x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        # x = torch.sigmoid(x)
        return x, z, ave, log_dev

    def encode(self, x):
        for m in self.encoder:
            x = m(x)

        # for VAE
        ave = self.lr_ave(x)  # average
        log_dev = self.lr_dev(x)  # log(sigma^2)

        z = self.seed_sampling(ave, log_dev)

        return z, ave, log_dev

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        # x = torch.sigmoid(x)
        return x


class CVAE(nn.Module):

    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, class_num):
        super(CVAE, self).__init__()

        self._set_random_seed(42)
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.lr_dev = nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1])
        self.lr_ave = nn.Linear(encoder_hidden_dims[-2], encoder_hidden_dims[-1])
        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_layers = []
        for i in range(len(encoder_hidden_dims)-2):
            if i == 0:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i]+class_num, encoder_hidden_dims[i+1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i+1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))
                # encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i+1]))

        # for visualization
        encoder_layers.append(self.lr_dev)
        encoder_layers.append(self.lr_ave)
        self.encoder = nn.ModuleList(encoder_layers)

        encoder_layers = encoder_layers[:-2]
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1]+class_num, decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
            elif i == len(decoder_hidden_dims)-1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
                # decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))
                # decoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i]))

        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder)
        print("Various layer: ", self.lr_dev, self.lr_ave)
        print(self.decoder)

    def _set_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    def _cat_xy(self, x, labels, dim=1):
        # x = x.to(self.device).view(-1, 28*28).to(torch.float32)
        # labels = F.one_hot(labels, num_classes=10).float().to(self.device)
        data = torch.cat([x, labels], dim=dim)
        data = data.cuda()

        return data

    def sampling(self, ave, log_dev):
        std = torch.exp(0.5 * log_dev)
        eps = torch.randn_like(std, memory_format=torch.contiguous_format)
        return eps.mul(std).add_(ave)  # return z sample

    def seed_sampling(self, ave, log_dev, seed=42):
        std = torch.exp(0.5 * log_dev)
        torch.manual_seed(seed)
        eps = torch.randn_like(std, memory_format=torch.contiguous_format)
        return eps.mul(std).add_(ave)  # return z sample

    def forward(self, x, y):
        x = self._cat_xy(x, y)

        for m in self.encoder:
            x = m(x)

        # for VAE
        ave = self.lr_ave(x)  # average
        log_dev = self.lr_dev(x)  # log(sigma^2)

        z = self.sampling(ave, log_dev)

        x = z
        x = self._cat_xy(x, y)

        for m in self.decoder:
            x = m(x)

        x = torch.sigmoid(x)

        return x, z, ave, log_dev

    def encode(self, x, y):
        x = self._cat_xy(x, y)
        for m in self.encoder:
            x = m(x)

        # for VAE
        ave = self.lr_ave(x)  # average
        log_dev = self.lr_dev(x)  # log(sigma^2)

        z = self.seed_sampling(ave, log_dev)  # hidden layer

        return z, ave, log_dev

    def decode(self, x, y):
        x = self._cat_xy(x, y)
        for m in self.decoder:
            x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        x = torch.sigmoid(x)
        return x
