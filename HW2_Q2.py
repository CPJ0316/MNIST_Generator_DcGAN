import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as dset
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.utils as vutils

def initial(self):
    self.device=torch.device('cpu')
    self.Gmodel=torch.load("./netG_ver5.pth", map_location=self.device)
    self.Dmodel=torch.load("./netD_ver5.pth", map_location=self.device)
    self.data_transforms =transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(contrast=(1.2, 1.5)),
        transforms.RandomRotation(degrees=60),#隨機旋轉 -30° 到 30° 之間
        transforms.ToTensor(),#將 PIL 圖像或 NumPy 陣列轉換為 PyTorch 的張量（Tensor）。
        transforms.Normalize((0.5,), (0.5,)), 
    ])
    self.no_data_transforms =transforms.Compose([
        transforms.ToTensor(),#將 PIL 圖像或 NumPy 陣列轉換為 PyTorch 的張量（Tensor）。
    ])
    all_dataset=dset.ImageFolder(root=self.loadFiles, transform=self.data_transforms)
    self.Data_loader = torch.utils.data.DataLoader(all_dataset, batch_size=64,shuffle=True, num_workers=1)
    no_dataset=dset.ImageFolder(root=self.loadFiles, transform=self.no_data_transforms)
    self.n_Data_loader = torch.utils.data.DataLoader(no_dataset, batch_size=64,shuffle=True, num_workers=1)

def show_augmentation(self):
    batch = next(iter(self.n_Data_loader))
    images, labels = batch  # 拆分为图像和标签
    # 对图像动态应用增强
    augmented_images = torch.stack([self.data_transforms(transforms.ToPILImage()(img)) for img in images])
    # 建造subgraph
    # 创建一个 16x8 的整体画布，但限制每个子图为 8x8
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))  # 1行2列，整体宽16高8

    # 原始数据集
    axes[0].imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axes[0].axis("off")
    axes[0].set_title("Training Dataset (Original)")

    # 增强后的数据集
    axes[1].imshow(np.transpose(vutils.make_grid(augmented_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axes[1].axis("off")
    axes[1].set_title("Training Dataset (Augmented)")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()
    
def show_structure(G,D):
    print(G)
    print()
    print(D)
    
def show_loss(loss):
    image = Image.open(loss).convert("RGB")
    fig, ax = plt.subplots()# 設置繪圖區域
    ax.imshow(image)# 顯示圖片
    ax.axis("off")  # 去除軸標籤
    plt.tight_layout()# 使佈局緊湊
    plt.show()  # 顯示圖片
    
def show_product_image(self):
    #image = Image.open(result_img).convert("RGB")
    #fig, ax = plt.subplots()# 設置繪圖區域
    #ax.imshow(image)# 顯示圖片
    #ax.axis("off")  # 去除軸標籤
    #plt.tight_layout()# 使佈局緊湊
    #plt.show()  # 顯示圖片

    # 1. 提取真實影像
    real_batch = next(iter(self.Data_loader))  # 從 DataLoader 中提取一個批次的影像
    real_images = real_batch[0].to(self.device)[:64]  # 選取批次中的前 64 張真實影像

    # 2. 使用生成器生成假影像
    latent_vectors = torch.randn(64, 100, 1, 1, device=self.device)  # 隨機生成 64 個 latent vectors
    fake_images = self.Gmodel(latent_vectors).detach().cpu()  # 通過生成器生成假影像，並從 GPU 移到 CPU

    # 3. 畫出並保存真實與假影像的對比
    plt.figure(figsize=(15, 15))

    # (a) 真實影像
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_images,  # 真實影像張量
                padding=5,
                normalize=True,  # 正規化到 [0, 1]
            ).cpu(),
            (1, 2, 0),
        )
    )

    # (b) 假影像
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                fake_images,  # 假影像張量
                padding=5,
                normalize=True,  # 正規化到 [0, 1]
            ),
            (1, 2, 0),
        )
    )

    # 4. 保存圖片
    #output_path = r"D:\CPJ\courses\1131\CvDL\Q2_git\real_fake_ver2.png"
    #plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #print(f"圖片已保存至 {output_path}")

    # 5. 顯示圖片
    plt.show()
