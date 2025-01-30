import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Model Hyperparameters
BATCH_SIZE = 128
LATENT_DIM = 64
CODEBOOK_SIZE = 512
EPOCHS = 1000
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Vector Quantization Layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # Flatten z
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # Compute distances
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight ** 2).sum(dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.T)

        # Get closest embeddings
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(z.shape)
        return quantized, indices


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


# VQ-VAE Model
class VQVAE(nn.Module):
    def __init__(self, latent_dim, num_embeddings):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantized


def main(model_path="vqvae.pt", visualize=True):
    # Training
    model = VQVAE(LATENT_DIM, CODEBOOK_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for images, _ in dataloader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon_images, _ = model(images)
            loss = F.mse_loss(recon_images, images)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

        if epoch % 100 == 0:
            # # 結果の可視化
            if visualize:
                model.eval()
                with torch.no_grad():
                    images, _ = next(iter(dataloader))
                    images = images.to(DEVICE)
                    recon_images, _ = model(images)

                # Plot Original and Reconstructed Images
                fig, axes = plt.subplots(2, 10, figsize=(10, 2))
                for i in range(10):
                    axes[0, i].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    axes[0, i].axis('off')
                    axes[1, i].imshow(np.transpose(recon_images[i].cpu().numpy(), (1, 2, 0)))
                    axes[1, i].axis('off')
                plt.show()

            # モデルの保存
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")


if __name__ == "__main__":
    main()
