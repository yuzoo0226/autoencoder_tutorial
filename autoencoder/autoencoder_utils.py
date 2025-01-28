#!/usr/bin/env python3

import os
import time
import random
import argparse
import numpy as np
from icecream import ic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model import Autoencoder, VAE, CVAE
import matplotlib.pyplot as plt


class AutoencoderUtils():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed(42)

        self.encoder_hidden_dims = args.encoder_dims
        self.decoder_hidden_dims = args.decoder_dims
        self.lr = args.lr

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """
        ベクトルのコサイン類似度を計算する関数
        Args:
            vec1 (numpy.ndarray): ベクトル1
            vec2 (numpy.ndarray): ベクトル2
        Returns:
            float: コサイン類似度
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    @staticmethod
    def _load_pt_file(file_path: str):
        # ic(file_path)
        data = torch.load(file_path)
        data = data.to(torch.float32)
        return data

    def set_random_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_pt_dir(self, pts_dir: str):

        # フォルダ内のファイル名を取得して名前順にソート
        file_names = sorted(
            [f for f in os.listdir(pts_dir) if f.lower().endswith(('.pt'))]
        )

        data_list = []

        for file_name in file_names:
            data = self._load_pt_file(file_path=os.path.join(pts_dir, file_name))
            data_list.append(data)

        data = torch.cat(data_list, dim=0)
        data = data.to(torch.float32)

        return data

    def train(self, num_epochs=10, checkpoint_path="checkpoint.pt", pts_dir="../pts/"):

        model = Autoencoder(self.encoder_hidden_dims, self.decoder_hidden_dims).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        inputs = self._load_pt_dir(pts_dir)
        inputs = inputs.to(torch.float32)

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0

            inputs = inputs.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            ic(f"Epoch [{epoch}/{num_epochs}], Training Loss: {running_loss:.5f}")

        ic("Training Finished")

        checkpoint_params = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss,
        }
        torch.save(checkpoint_params, checkpoint_path)


class VAEUtils():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed(42)

        self.encoder_hidden_dims = args.encoder_dims
        self.decoder_hidden_dims = args.decoder_dims
        self.lr = args.lr
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.eps = np.spacing(1)
        self.save_iteration = args.save_iteration

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """
        ベクトルのコサイン類似度を計算する関数
        Args:
            vec1 (numpy.ndarray): ベクトル1
            vec2 (numpy.ndarray): ベクトル2
        Returns:
            float: コサイン類似度
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def set_random_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def test_mnist(self):
        from torch.utils.data import DataLoader
        from torchvision.datasets import MNIST, FashionMNIST
        import torchvision.transforms as transforms
        BATCH_SIZE = 2000
        num_epochs = 20000

        trainval_data = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())

        train_size = int(len(trainval_data) * 0.8)
        val_size = int(len(trainval_data) * 0.2)
        train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        print("train data size: ", len(train_data))  # train data size:  48000
        print("train iteration number: ", len(train_data)//BATCH_SIZE)  # train iteration number:  480
        print("val data size: ", len(val_data))  # val data size:  12000
        print("val iteration number: ", len(val_data)//BATCH_SIZE)  # val iteration number:  120

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VAE(self.encoder_hidden_dims, self.decoder_hidden_dims).to(device)
        # model = VAE(self.encoder_hidden_dims, self.decoder_hidden_dims).to(device)

        # build model
        # model = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
        if torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
        history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels": []}

        for epoch in range(num_epochs):
            model.train()
            for i, (x, labels) in enumerate(train_loader):
                input = x.to(self.device).view(-1, 28*28).to(torch.float32)
                output, z, ave, log_dev = model(input)

                history["ave"].append(ave)
                history["log_dev"].append(log_dev)
                history["z"].append(z)
                history["labels"].append(labels)
                loss = self.ce_criterion(output, input, ave, log_dev)
                # loss = self.loss_function(output, input, ave, log_dev)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 5 == 0:
                    print(f'Epoch: {epoch+1}, loss: {loss/len(x): 0.4f}')
                    history["train_loss"].append(loss/len(x))

            if epoch % 100 == 0:
                ave_tensor = torch.stack(history["ave"])
                log_var_tensor = torch.stack(history["log_dev"])
                z_tensor = torch.stack(history["z"])
                labels_tensor = torch.stack(history["labels"])
                print(ave_tensor.size())  # torch.Size([9600, 100, 2])
                print(log_var_tensor.size())  # torch.Size([9600, 100, 2])
                print(z_tensor.size())  # torch.Size([9600, 100, 2])
                print(labels_tensor.size())  # torch.Size([9600, 100])

                ave_np = ave_tensor.to('cpu').detach().numpy().copy()
                log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
                z_np = z_tensor.to('cpu').detach().numpy().copy()
                labels_np = labels_tensor.to('cpu').detach().numpy().copy()
                print(ave_np.shape)  # (9600, 100, 2)
                print(log_var_np.shape)  # (9600, 100, 2)
                print(z_np.shape)  # (9600, 100, 2)
                print(labels_np.shape)  # (9600, 100)

                checkpoint_params = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }
                torch.save(checkpoint_params, "test_mnist.pt")

                def get_hidden_dim():
                    model.eval()
                    z_list = []
                    labels_list = []
                    with torch.no_grad():
                        for data, labels in val_loader:
                            data = data.to(self.device).view(-1, 28*28).to(torch.float32)
                            data = data.cuda()
                            recon, z, mu, log_var = model(data)
                            z_list.append(z)
                            labels_list.append(labels)

                        z = torch.cat(z_list, dim=0)
                        labels = torch.cat(labels_list, dim=0)
                        return z, labels

                z, labels = get_hidden_dim()

                z_np = z.to('cpu').detach().numpy()
                labels_np = labels.to('cpu').detach().numpy()

                cmap_keyword = "tab10"
                cmap = plt.get_cmap(cmap_keyword)

                plt.figure(figsize=[10, 10])
                for label in range(10):
                    x = []
                    y = []
                    for idx, estimate_label in enumerate(labels_np):
                        if estimate_label == label:
                            x.append(z_np[idx][0])
                            y.append(z_np[idx][1])
                    plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
                    plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
                plt.legend(loc="upper left")
                plt.show()

    def mse_criterion(self, predict, target, ave, log_dev):
        mse_loss = self.mse_loss(predict, target)
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = mse_loss + kl_loss
        return loss

    def ce_criterion(self, predict, target, ave, log_dev):
        ce_loss = self.ce_loss(predict, target)
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = ce_loss + kl_loss
        return loss

    def bce_criterion(self, predict, target, ave, log_dev):
        bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = bce_loss + kl_loss
        return loss

    @staticmethod
    def get_indices_out_of_bounds(tensor):
        # 1以上または-1以下の値をもつインデックスを取得
        indices = torch.nonzero((tensor > 1) | (tensor < 0), as_tuple=True)[0]
        return indices


class CVAEUtils():
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed(42)

        self.encoder_hidden_dims = args.encoder_dims
        self.decoder_hidden_dims = args.decoder_dims
        self.lr = args.lr
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.eps = np.spacing(1)
        self.save_iteration = args.save_iteration

    def set_random_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def mse_criterion(self, predict, target, ave, log_dev):
        mse_loss = self.mse_loss(predict, target)
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = mse_loss + kl_loss
        return loss

    def ce_criterion(self, predict, target, ave, log_dev):
        ce_loss = self.ce_loss(predict, target)
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = ce_loss + kl_loss
        return loss

    def bce_criterion(self, predict, target, ave, log_dev):
        bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = bce_loss + kl_loss
        return loss

    def get_hidden_dim(self, model, val_loader):
        model.eval()
        z_list = []
        labels_list = []
        with torch.no_grad():
            for x, labels in val_loader:
                inputs = x.to(self.device).view(-1, 28*28).to(torch.float32)
                labels = F.one_hot(labels, num_classes=10).float().to(self.device)
                recon, z, mu, log_var = model(inputs, labels)
                z_list.append(z)
                labels_list.append(labels)

            z = torch.cat(z_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            return z, labels

    def test_mnist(self):
        from torch.utils.data import DataLoader
        from torchvision.datasets import MNIST, FashionMNIST
        import torchvision.transforms as transforms
        BATCH_SIZE = 4000
        num_epochs = 20000

        trainval_data = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        os.mkdir("test_mnist/")

        train_size = int(len(trainval_data) * 0.8)
        val_size = int(len(trainval_data) * 0.2)
        train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        print("train data size: ", len(train_data))  # train data size:  48000
        print("train iteration number: ", len(train_data)//BATCH_SIZE)  # train iteration number:  480
        print("val data size: ", len(val_data))  # val data size:  12000
        print("val iteration number: ", len(val_data)//BATCH_SIZE)  # val iteration number:  120

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CVAE(self.encoder_hidden_dims, self.decoder_hidden_dims, 10).to(device)

        # build model
        # model = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
        if torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
        history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels": []}

        for epoch in range(1, num_epochs):
            model.train()
            for i, (x, labels) in enumerate(train_loader):
                inputs = x.view(-1, 28*28).to(self.device)
                labels = F.one_hot(labels, num_classes=10).float().to(self.device)
                outputs, z, ave, log_dev = model(inputs, labels)

                history["ave"].append(ave)
                history["log_dev"].append(log_dev)
                history["z"].append(z)
                history["labels"].append(labels)
                loss = self.ce_criterion(outputs, inputs, ave, log_dev)
                # loss = self.loss_function(output, input, ave, log_dev)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 5 == 0:
                    print(f'Epoch: {epoch}, loss: {loss/len(x): 0.4f}')
                    history["train_loss"].append(loss/len(x))

            if epoch % self.save_iteration == 0:
                ave_tensor = torch.stack(history["ave"])
                log_var_tensor = torch.stack(history["log_dev"])
                z_tensor = torch.stack(history["z"])
                labels_tensor = torch.stack(history["labels"])
                print(ave_tensor.size())  # torch.Size([9600, 100, 2])
                print(log_var_tensor.size())  # torch.Size([9600, 100, 2])
                print(z_tensor.size())  # torch.Size([9600, 100, 2])
                print(labels_tensor.size())  # torch.Size([9600, 100])

                ave_np = ave_tensor.to('cpu').detach().numpy().copy()
                log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
                z_np = z_tensor.to('cpu').detach().numpy().copy()
                labels_np = labels_tensor.to('cpu').detach().numpy().copy()
                print(ave_np.shape)  # (9600, 100, 2)
                print(log_var_np.shape)  # (9600, 100, 2)
                print(z_np.shape)  # (9600, 100, 2)
                print(labels_np.shape)  # (9600, 100)

                checkpoint_params = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }
                torch.save(checkpoint_params, f"test_mnist/test_mnist_cvae_epoch_{epoch}.pt")

                z, labels = self.get_hidden_dim(model=model, val_loader=val_loader)

                z_np = z.to('cpu').detach().numpy()
                labels_np = labels.to('cpu').detach().numpy()

                cmap_keyword = "tab10"
                cmap = plt.get_cmap(cmap_keyword)

                plt.figure(figsize=[10, 10])
                for label in range(10):
                    x = []
                    y = []
                    for idx, estimate_label_vector in enumerate(labels_np):
                        estimate_label = np.argmax(estimate_label_vector, axis=0)
                        # ic(estimate_label_vector, estimate_label)
                        if estimate_label == label:
                            x.append(z_np[idx][0])
                            y.append(z_np[idx][1])
                    plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
                    plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
                plt.legend(loc="upper left")
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "-m", type=str, choices=["ae", "vae", "cvae"], default="vae")
    parser.add_argument("--num_epochs", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=0.01)
    # parser.add_argument("--encoder_dims", nargs="+", type=int, default=[512, 256, 128, 64, 32, 2])
    # parser.add_argument("--decoder_dims", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--encoder_dims", nargs="+", type=int, default=[28*28, 512, 256, 2])
    parser.add_argument("--decoder_dims", nargs="+", type=int, default=[256, 512, 28*28])
    parser.add_argument("--checkpoint_path", type=str, default="../test_mnist.pt")
    parser.add_argument("--save_iteration", type=int, default=10)
    parser.add_argument("--pts_dir", type=str, default="../")
    args = parser.parse_args()

    if args.base_model.lower() == "vae":
        utils = VAEUtils(args)
        utils.set_random_seed(42)
        # utils.train(num_epochs=args.num_epochs, checkpoint_path=args.checkpoint_path)
        utils.test_mnist()

    elif args.base_model.lower() == "cvae":
        utils = CVAEUtils(args)
        utils.set_random_seed(42)
        utils.test_mnist()

    else:
        utils = AutoencoderUtils(args)
        utils.set_random_seed(42)
