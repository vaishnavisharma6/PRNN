
import kagglehub
kagglehub.login()


aptos2019_blindness_detection_path = kagglehub.competition_download('aptos2019-blindness-detection')
abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torchvision.models import inception_v3, Inception_V3_Weights

from scipy.linalg import sqrtm
from tqdm import tqdm


device = "cuda"
PLANTVILLAGE_DIR = os.path.join(
    abdallahalidev_plantvillage_dataset_path,
    "plantvillage dataset",
    "color"
)

APTOS_IMG_DIR = aptos2019_blindness_detection_path

train_csv_path = os.path.join(
    aptos2019_blindness_detection_path,
    "train.csv"
)

test_csv_path = os.path.join(
    aptos2019_blindness_detection_path,
    "test.csv"
)

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(train_df.head())
print(test_df.head())

train_img_dir = os.path.join(
    aptos2019_blindness_detection_path,
    "train_images"
)

test_img_dir = os.path.join(
    aptos2019_blindness_detection_path,
    "test_images"
)

print(os.listdir(train_img_dir)[:5])
APTOS_CSV = os.path.join(aptos2019_blindness_detection_path, "train.csv")
APTOS_IMG_DIR = os.path.join(aptos2019_blindness_detection_path, "train_images")

OUTPUT_DIR = "/kaggle/working/phase3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================

IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS_VAE = 15
EPOCHS_GAN = 20
LR = 2e-4
BETA1 = 0.5
NUM_WORKERS = 2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# DATASETS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

plant_dataset = datasets.ImageFolder(
    root=PLANTVILLAGE_DIR,
    transform=transform
)

plant_loader = DataLoader(
    plant_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("PlantVillage images:", len(plant_dataset))


class AptosClassDataset(Dataset):
    def __init__(self, csv_path, img_dir, target_class=4, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["diagnosis"] == target_class].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id_code"]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


aptos_class4_dataset = AptosClassDataset(
    APTOS_CSV,
    APTOS_IMG_DIR,
    target_class=4,
    transform=transform
)

aptos_loader = DataLoader(
    aptos_class4_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("APTOS Class 4 images:", len(aptos_class4_dataset))

# ============================================================
# 3.7 CONVOLUTIONAL VAE
# ============================================================

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / x.size(0)

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / x.size(0)

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


vae = ConvVAE(LATENT_DIM).to(device)
vae_opt = optim.Adam(vae.parameters(), lr=LR)

vae_recon_losses = []
vae_kl_losses = []

for epoch in range(EPOCHS_VAE):
    vae.train()
    total_recon = 0
    total_kl = 0

    for x, _ in tqdm(plant_loader, desc=f"VAE Epoch {epoch+1}/{EPOCHS_VAE}"):
        x = x.to(device)

        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)

        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()

        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    avg_recon = total_recon / len(plant_loader)
    avg_kl = total_kl / len(plant_loader)

    vae_recon_losses.append(avg_recon)
    vae_kl_losses.append(avg_kl)

    print(f"Epoch [{epoch+1}/{EPOCHS_VAE}] Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_plantvillage.pth"))

plt.figure(figsize=(8, 5))
plt.plot(vae_recon_losses, label="Reconstruction Term")
plt.plot(vae_kl_losses, label="Regularization KL Term")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE ELBO Terms")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "vae_elbo_terms.png"))
plt.show()

# ============================================================
# VAE SAMPLE GRID
# ============================================================

vae.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM).to(device)
    vae_samples = vae.decode(z).cpu()

grid = make_grid(vae_samples, nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("VAE Generated Leaf Images")
plt.show()

save_image(vae_samples, os.path.join(OUTPUT_DIR, "vae_generated_grid.png"), nrow=4, normalize=True)

# ============================================================
# 3.8 DCGAN ON PLANTVILLAGE
# ============================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=128, use_batchnorm=True):
        super().__init__()

        layers = []

        layers += [
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False)
        ]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(512)]
        layers += [nn.ReLU(True)]

        layers += [
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        ]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(256)]
        layers += [nn.ReLU(True)]

        layers += [
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        ]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(128)]
        layers += [nn.ReLU(True)]

        layers += [
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        ]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(64)]
        layers += [nn.ReLU(True)]

        layers += [
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


def train_dcgan(loader, epochs, use_batchnorm=True, save_name="dcgan"):
    G = Generator(LATENT_DIM, use_batchnorm=use_batchnorm).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1).to(device)

    G_losses = []
    D_losses = []

    for epoch in range(20):
        G.train()
        D.train()

        total_G = 0
        total_D = 0

        for real, _ in tqdm(loader, desc=f"{save_name} Epoch {epoch+1}/{epochs}"):
            real = real.to(device)
            bsz = real.size(0)

            real_labels = torch.ones(bsz).to(device)
            fake_labels = torch.zeros(bsz).to(device)

            # --------------------
            # Train Discriminator
            # --------------------
            noise = torch.randn(bsz, LATENT_DIM, 1, 1).to(device)
            fake = G(noise)

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D_real = criterion(D_real, real_labels)
            loss_D_fake = criterion(D_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --------------------
            # Train Generator
            # --------------------
            D_fake_for_G = D(fake)
            loss_G = criterion(D_fake_for_G, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_D += loss_D.item()
            total_G += loss_G.item()

        avg_D = total_D / len(loader)
        avg_G = total_G / len(loader)

        D_losses.append(avg_D)
        G_losses.append(avg_G)

        print(f"Epoch [{epoch+1}/{epochs}] D Loss: {avg_D:.4f}, G Loss: {avg_G:.4f}")

        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_image(samples, os.path.join(OUTPUT_DIR, f"{save_name}_epoch_{epoch+1}.png"),
                       nrow=4, normalize=True)

    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, f"{save_name}_generator.pth"))
    torch.save(D.state_dict(), os.path.join(OUTPUT_DIR, f"{save_name}_discriminator.pth"))

    return G, D, G_losses, D_losses


G_plant, D_plant, G_losses, D_losses = train_dcgan(
    plant_loader,
    EPOCHS_GAN,
    use_batchnorm=True,
    save_name="dcgan_plantvillage"
)

plt.figure(figsize=(8, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCGAN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "dcgan_losses.png"))
plt.show()

# ============================================================
# DCGAN SAMPLE GRID


G_plant.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    gan_samples = G_plant(z).cpu()

grid = make_grid(gan_samples, nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("DCGAN Generated Leaf Images")
plt.show()

save_image(gan_samples, os.path.join(OUTPUT_DIR, "dcgan_generated_grid.png"), nrow=4, normalize=True)

# ============================================================
# 3.9 TRAIN DCGAN ON APTOS CLASS 4


G_aptos, D_aptos, G_aptos_losses, D_aptos_losses = train_dcgan(
    aptos_loader,
    EPOCHS_GAN,
    use_batchnorm=True,
    save_name="dcgan_aptos_class4"
)

G_aptos.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    aptos_samples = G_aptos(z).cpu()

grid = make_grid(aptos_samples, nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("APTOS Class 4 DCGAN Samples")
plt.show()

save_image(aptos_samples, os.path.join(OUTPUT_DIR, "aptos_class4_dcgan_grid.png"), nrow=4, normalize=True)

# ============================================================
# FID IMPLEMENTATION USING INCEPTION-V3 + SCIPY


fid_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        self.model = model

    def forward(self, x):
        return self.model(x)


inception = InceptionFeatureExtractor().to(device)
inception.eval()


def denormalize(x):
    return (x * 0.5) + 0.5


def get_inception_features_from_loader(loader, max_images=500):
    features = []
    count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting real features"):
            imgs = imgs.to(device)
            imgs = denormalize(imgs)
            imgs = torch.stack([fid_transform(img) for img in imgs])

            feats = inception(imgs)
            features.append(feats.cpu().numpy())

            count += imgs.size(0)
            if count >= max_images:
                break

    features = np.concatenate(features, axis=0)
    return features[:max_images]


def get_inception_features_from_generator(G, max_images=500, batch_size=64):
    features = []
    count = 0
    G.eval()

    with torch.no_grad():
        while count < max_images:
            bsz = min(batch_size, max_images - count)
            z = torch.randn(bsz, LATENT_DIM, 1, 1).to(device)
            imgs = G(z)

            imgs = denormalize(imgs)
            imgs = torch.stack([fid_transform(img) for img in imgs])

            feats = inception(imgs)
            features.append(feats.cpu().numpy())

            count += bsz

    features = np.concatenate(features, axis=0)
    return features[:max_images]


def calculate_fid(real_features, fake_features):
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)

    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2

    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


real_features = get_inception_features_from_loader(aptos_loader, max_images=500)
fake_features = get_inception_features_from_generator(G_aptos, max_images=500)

fid_score = calculate_fid(real_features, fake_features)

print("Exact FID score:", fid_score)

with open(os.path.join(OUTPUT_DIR, "fid_score.txt"), "w") as f:
    f.write(f"FID Score: {fid_score}\n")

# ============================================================
# 3.10 REMOVE BATCH NORMALIZATION FROM APTOS GENERATOR


G_aptos_no_bn, D_aptos_no_bn, G_no_bn_losses, D_no_bn_losses = train_dcgan(
    aptos_loader,
    EPOCHS_GAN,
    use_batchnorm=False,
    save_name="dcgan_aptos_no_batchnorm"
)

G_aptos_no_bn.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    no_bn_samples = G_aptos_no_bn(z).cpu()

grid = make_grid(no_bn_samples, nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("APTOS Generator Without BatchNorm")
plt.show()

save_image(no_bn_samples, os.path.join(OUTPUT_DIR, "aptos_no_batchnorm_grid.png"), nrow=4, normalize=True)

plt.figure(figsize=(8, 5))
plt.plot(G_aptos_losses, label="Generator With BatchNorm")
plt.plot(G_no_bn_losses, label="Generator Without BatchNorm")
plt.xlabel("Epoch")
plt.ylabel("Generator Loss")
plt.title("Effect of Removing BatchNorm")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "batchnorm_ablation_loss.png"))
plt.show()



