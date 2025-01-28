#!/usr/bin/env python3

import os
import sys
import cv2
import time
import copy
import random
import argparse
import numpy as np
from icecream import ic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import dearpygui.dearpygui as dpg
from typing import Optional, List, Tuple
from autoencoder.model import Autoencoder, VAE, CVAE



class CVAEVisualizer():
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

        self.image_show_size = 280*2
        self.calss_num = 10

        self.encoder_hidden_dims = args.encoder_dims
        self.decoder_hidden_dims = args.decoder_dims

        self.cvae_model = CVAE(self.encoder_hidden_dims, self.decoder_hidden_dims, self.calss_num).to(self.device)
        self.cvae_model._set_random_seed(42)
        cvae_checkpoint_path = args.checkpoint_path
        self.ae_checkpoint = torch.load(cvae_checkpoint_path)
        self.cvae_model.load_state_dict(self.ae_checkpoint['model_state_dict'])
        self.cvae_model.eval()
        self.cvae_model = self.cvae_model.to(self.device)

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

    @staticmethod
    def _resize_image(image, max_length=500):
        # h, w を取得
        h, w = image.shape[:2]

        # h と w のうち大きい方を取得
        max_dim = max(h, w)

        # 大きい方の長さが100になるスケールを計算
        scale = max_length / max_dim

        # そのスケール値を使用してリサイズ
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized_image, scale

    def _set_image_to_dynamic_texture(self, image, tag="image_texture"):
        self.current_view_cv = copy.deepcopy(image)
        image, _ = self._resize_image(image, self.image_show_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        dpg.set_value(tag, image/255.0)

    def _generate_image(self, label, inputs=None):
        # label を torch.Tensor に変換
        label_tensor = torch.tensor(label, dtype=torch.long)

        # one-hot エンコード
        label = F.one_hot(label_tensor, num_classes=10).float().to(self.device)

        # 以下は既存の処理
        z = torch.randn(1, 2).to(self.device)
        generated_image = self.cvae_model.decode(z, label)
        generated_image = generated_image.view(28, 28).cpu().detach().numpy()
        scaled_image = (generated_image * 255).astype(np.uint8)
        # ic(scaled_image)

        cv2.imwrite("test.png", scaled_image)
        return scaled_image

    def generate_image_callback(self, sender, app_data):
        label = int(dpg.get_value("condition_label"))
        generated_image = self._generate_image(label=[label])
        self._set_image_to_dynamic_texture(generated_image)

    def train_mnist_callback(self, sender, app_data):
        from torch.utils.data import DataLoader
        from torchvision.datasets import MNIST, FashionMNIST
        import torchvision.transforms as transforms
        BATCH_SIZE = 4000
        num_epochs = 20000

        trainval_data = MNIST("./autoencoder/data", train=True, download=True, transform=transforms.ToTensor())

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

        for epoch in range(num_epochs):
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

                torch.save(checkpoint_params, f"test_mnist_cvae_gui_epoch{epoch}.pt")

                # z, labels = self.get_hidden_dim(model=model, val_loader=val_loader)

                # z_np = z.to('cpu').detach().numpy()
                # labels_np = labels.to('cpu').detach().numpy()

                # cmap_keyword = "tab10"
                # cmap = plt.get_cmap(cmap_keyword)

                # plt.figure(figsize=[10, 10])
                # for label in range(10):
                #     x = []
                #     y = []
                #     for idx, estimate_label_vector in enumerate(labels_np):
                #         estimate_label = np.argmax(estimate_label_vector, axis=0)
                #         # ic(estimate_label_vector, estimate_label)
                #         if estimate_label == label:
                #             x.append(z_np[idx][0])
                #             y.append(z_np[idx][1])
                #     plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
                #     plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
                # plt.legend(loc="upper left")
                # plt.show()

    def setup_gui(self) -> None:
        """
        Set up the DearPyGui interface.
        """
        dpg.create_context()
        dpg.create_viewport(title='CVAE-based MNIST Image Generator', width=600, height=630)

        blank_image = np.zeros((self.image_show_size, self.image_show_size, 4), dtype=np.uint8)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(width=self.image_show_size, height=self.image_show_size, default_value=blank_image, tag="image_texture")

        with dpg.window(label="Image Processing GUI"):
            # if self.debug:
            #     # dpg.add_input_text(label="Image Path", default_value="/mnt/home/yuga-y/usr/splat_ws/datasets/open_gaussian/ramen/images/frame_00001.jpg", tag="image_path")
            #     dpg.add_input_text(label="Image Path", default_value="/mnt/home/yuga-y/usr/splat_ws/datasets/seg_any_gaussian/waldo_kitchen/images/frame_00089.jpg", tag="image_path")
            #     with dpg.group(horizontal=True):
            #         dpg.add_button(label="Load Image", callback=self.load_image)
            #         dpg.add_button(label="Perform Segmentation", callback=self.segment_image)
            #         dpg.add_button(label="Extract and Plot Features", callback=self.extract_feature)
            #     dpg.add_text("Load Directory")

            # dpg.add_input_text(label="Image Root Directory", default_value="/mnt/home/yuga-y/usr/splat_ws/datasets/seg_any_gaussian/waldo_kitchen/images_temp/", tag="image_root_dir")
            # with dpg.group(horizontal=True):
            #     dpg.add_button(label="Load Images(Dir)", callback=self.load_images)
            #     dpg.add_button(label="Perform Segmentation(Dir)", callback=self.segment_images)
            #     dpg.add_button(label="Extract and Plot Features(Dir)", callback=self.extract_features)
            #     dpg.add_button(label="Dump results", callback=self.dump_results_callback)

            # dpg.add_combo(label="Choose an frame", items=self.frame_indexes, callback=self.frame_select_callback, tag="frame_select")

            with dpg.group(horizontal=True):
                with dpg.group(horizontal=False):
                    dpg.add_image("image_texture", label="loaded image", tag="dynamic_image")
                    # self.text_current_frame_widget_id = dpg.add_text("frame_name", tag="frame_name")
                # with dpg.plot(label="Feature Plot", width=800, height=800, query=True):
                #     dpg.add_plot_legend()  # 凡例追加
                #     dpg.add_plot_axis(dpg.mvXAxis, label='x', tag='xaxis')
                #     dpg.add_plot_axis(dpg.mvYAxis, label='y', tag='yaxis')

                #     dpg.add_scatter_series([0, 1], [1, 2], label='Vison feature points', parent=dpg.last_item(), tag='feature')
                #     dpg.add_draw_layer(tag="roi_layer", parent="feature_plot")
                #     dpg.add_draw_layer(tag="target_feature_layer", parent="feature_plot")
                #     dpg.add_draw_layer(tag="target_image_layer", parent="feature_plot")
                #     dpg.add_draw_layer(tag='text_feature_layer', parent="feature_plot")

                #     with dpg.handler_registry():
                #         dpg.add_mouse_move_handler(callback=self.mouse_move_callback)
                #         dpg.add_mouse_click_handler(callback=self.click_callback)
                #         dpg.add_mouse_release_handler(callback=self.release_callback)

            with dpg.group(horizontal=True):
                # dpg.add_input_text(label="Output path", default_value="", tag="output_path")
                dpg.add_button(label="Generate New Image", callback=self.generate_image_callback)
                dpg.add_input_text(label="label", default_value="1", tag="condition_label")

            # dpg.add_button(label="Train", callback=self.train_mnist_callback)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self) -> None:
        """
        Run the DearPyGui application.
        """
        self.setup_gui()
        dpg.start_dearpygui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "-m", type=str, choices=["ae", "vae", "cvae"], default="vae")
    parser.add_argument("--num_epochs", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=0.01)
    # parser.add_argument("--encoder_dims", nargs="+", type=int, default=[512, 256, 128, 64, 32, 2])
    # parser.add_argument("--decoder_dims", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--encoder_dims", nargs="+", type=int, default=[28*28, 512, 256, 2])
    parser.add_argument("--decoder_dims", nargs="+", type=int, default=[256, 512, 28*28])
    parser.add_argument("--checkpoint_path", type=str, default="model/cvae_epoch_600.pt")
    parser.add_argument("--save_iteration", type=int, default=1000)
    parser.add_argument("--pts_dir", type=str, default="model/")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    app = CVAEVisualizer(args)
    app.run()
