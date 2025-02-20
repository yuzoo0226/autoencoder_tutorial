import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. **AutoEncoder モデル定義**
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # ピクセル値 [0,1] に制約
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z  # 再構成画像と埋め込みベクトルを返す


def visualize_embeddings(model, dataloader):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, embeddings = model(images)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Embeddings")
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 28x28 を 784次元ベクトルに変換
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    # 3. **モデル・損失関数・最適化関数**
    latent_dim = 32  # 埋め込み空間の次元
    model = AutoEncoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    margin_loss = nn.MarginRankingLoss(margin=1.0)

    # 4. **学習ループ**
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        total_margin_loss = 0

        for batch in train_loader:
            images, labels = batch
            labels = labels.to(device)
            images = images.to(device)

            optimizer.zero_grad()
            recon_images, embeddings = model(images)  # 再構成 & 埋め込み
            
            # MSE Loss（再構成誤差）
            loss_recon = mse_loss(recon_images, images)

            # Margin Ranking Loss の適用
            batch_size = embeddings.size(0)
            # idx1, idx2 = torch.randint(0, batch_size, (batch_size,)).to(device), torch.randint(0, batch_size, (batch_size,))
            idx1, idx2 = torch.randint(0, batch_size, (batch_size,)).to(device), torch.randint(0, batch_size, (batch_size,)).to(device)
            
            z1, z2 = embeddings[idx1], embeddings[idx2]  # 2つの異なるサンプル
            dist = torch.norm(z1 - z2, dim=1)  # ユークリッド距離
            
            # ラベルが同じなら +1, 異なるなら -1
            print(idx1)
            print(idx2)
            print(z1)
            print(z2)
            target = (labels[idx1] == labels[idx2]).float() * 2 - 1
            print(target)
            loss_margin = margin_loss(dist, torch.zeros_like(dist), target)  # 0 との比較
            print(loss_margin)

            # 合計損失
            loss = loss_recon + 0.1 * loss_margin  # 重みを調整
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_margin_loss += loss_margin.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Margin Loss: {total_margin_loss:.4f}")
    
    return model, train_dataset


if __name__ == "__main__":
    model, train_dataset = main()
    visualize_embeddings(model, DataLoader(train_dataset, batch_size=1000, shuffle=True))
