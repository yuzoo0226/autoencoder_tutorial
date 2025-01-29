import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# AutoEncoder の定義
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 入力: (3, 32, 32) → 出力: (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 入力: (16, 16, 16) → 出力: (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 入力: (32, 8, 8) → 出力: (64, 4, 4)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 入力: (64, 4, 4) → 出力: (32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 入力: (32, 8, 8) → 出力: (16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 入力: (16, 16, 16) → 出力: (3, 32, 32)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    epochs = 1
    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)

            # 順伝播
            outputs = model(images)
            loss = criterion(outputs, images)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 結果の可視化
    model.eval()
    dataiter = iter(dataloader)
    images, _ = next(dataiter)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # 可視化
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for i in range(6):
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, i].axis("off")

        axes[1, i].imshow(outputs[i].cpu().permute(1, 2, 0))
        axes[1, i].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
