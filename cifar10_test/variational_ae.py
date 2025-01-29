import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Variational AutoEncoder (VAE) の定義
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (3, 32, 32) → (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, 16, 16) → (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 8, 8) → (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64 * 4 * 4, 512)  # 2次元潜在空間
        self.fc_logvar = nn.Linear(64 * 4 * 4, 512)  # 2次元潜在空間
        self.fc_decode = nn.Linear(512, 64 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 4, 4) → (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 8, 8) → (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 16, 16) → (3, 32, 32)
            # nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = self.decoder(x)
        return x, z, mu, logvar


criterion = nn.MSELoss()
# ce_loss = nn.CrossEntropyLoss(reduction='mean')
ce_loss = nn.CrossEntropyLoss(reduction='sum')


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = ce_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def get_hidden_dim(model, data_loder, device="cuda"):
    model.eval()
    z_list = []
    labels_list = []
    with torch.no_grad():
        for data, labels in data_loder:
            images = data.to(device)
            outputs, z, mu, logvar = model(images)
            # data = data.to(device).view(-1, 28*28).to(torch.float32)
            # data = data.cuda()
            recon, z, mu, log_var = model(images)
            z_list.append(z)
            labels_list.append(labels)

        z = torch.cat(z_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return z, labels


def main():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 のロード
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # モデルの準備
    model = VAE().to(device)
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    epochs = 10000
    for epoch in range(1, epochs+1):
        for images, _ in dataloader:
            images = images.to(device)
            # print(images)

            # 順伝播
            outputs, z, mu, logvar = model(images)
            loss = vae_loss(outputs, images, mu, logvar)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], Loss: {(loss.item()/len(images)):.4f}")

        if epoch % 10 == 0:
            # # 結果の可視化
            # model.eval()
            # dataiter = iter(dataloader)
            # images, _ = next(dataiter)
            # images = images.to(device)
            # # print("============= inputs ===============")
            # # print(images)

            # with torch.no_grad():
            #     outputs, z, mu, logvar = model(images)
            #     # print("============== outputs ============")
            #     # print(outputs)

            # # 可視化
            # fig, axes = plt.subplots(2, 6, figsize=(10, 4))
            # for i in range(6):
            #     axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            #     axes[0, i].axis("off")

            #     axes[1, i].imshow(outputs[i].cpu().permute(1, 2, 0))
            #     axes[1, i].axis("off")
            # plt.show()

            # z, labels = get_hidden_dim(model, dataloader)

            # z_np = z.to('cpu').detach().numpy()
            # labels_np = labels.to('cpu').detach().numpy()

            # cmap_keyword = "tab10"
            # cmap = plt.get_cmap(cmap_keyword)

            # plt.figure(figsize=[10, 10])
            # for label in range(10):
            #     x = []
            #     y = []
            #     for idx, estimate_label in enumerate(labels_np):
            #         if estimate_label == label:
            #             x.append(z_np[idx][0])
            #             y.append(z_np[idx][1])
            #     plt.scatter(x, y, color=cmap(label/9), label=label, s=15)
            #     plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
            # plt.legend(loc="upper left")
            # plt.show()

            # モデルの保存
            torch.save(model.state_dict(), "vae_model.pth")
            print("Model saved as vae_model.pth")

            model.train()


if __name__ == "__main__":
    main()
