import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import models
import torchvision.transforms as transforms
import numpy as np

def initial(self):
    self.device=torch.device('cpu')
    self.model=torch.load("./vgg19_bn.pth", map_location=self.device)
    self.transform= transforms.Compose(
    [
    transforms.RandomRotation(degrees=60),#隨機旋轉 -30° 到 30° 之間
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def show_augmentation(self.files,self.transform):

def show_structure(self.model):
    
def show_loss(self.loss):
    
def show_product_image():
    