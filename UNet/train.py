import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import glob
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from tqdm import tqdm

device=torch.device('cuda')

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),# 为什么我们需要BatchNorm？
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class UNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(UNet,self).__init__()
        
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.left_block_1=conv_block(ch_in=img_ch,ch_out=64)
        self.left_block_2=conv_block(64,128)
        self.left_block_3=conv_block(128,256)
        self.left_block_4=conv_block(256,512)
        
        self.bottom_block=conv_block(512,1024)
        
        self.up4=up_conv(1024,512)
        self.up3=up_conv(512,256)
        self.up2=up_conv(256,128)
        self.up1=up_conv(128,64)
    
        
        self.right_block_4=conv_block(1024,512)
        self.right_block_3=conv_block(512,256)
        self.right_block_2=conv_block(256,128)
        self.right_block_1=conv_block(128,64)
        
        self.conv1x1=nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        # Left
        
        x1=self.left_block_1(x)
        x2=self.left_block_2(self.maxpool(x1))
        x3=self.left_block_3(self.maxpool(x2))
        x4=self.left_block_4(self.maxpool(x3))
        
        # Bottom 
        
        bottom=self.bottom_block(self.maxpool(x4))
        
        # Right
        
        # # test
        # d4=torch.cat((x4,self.up4(bottom)),dim=1)
        # d4=self.right_block_4(d4)
        
        # d3=torch.cat((x3,self.up3(d4)),dim=1)
        # d3=self.right_block_3(d3)
        
        # d2=torch.cat((x2,self.up2(d3)),dim=1)
        # d2=self.right_block_2(d2)
  
        # d1=torch.cat((x1,self.up1(d2)),dim=1)
        # d1=self.right_block_1(d1)
        # # /test
        x_cat4=torch.cat((x4,self.up4(bottom)),dim=1)
        d4=self.right_block_4(x_cat4)
        
        x_cat3=torch.cat((x3,self.up3(d4)),dim=1)
        d3=self.right_block_3(x_cat3)
        
        x_cat2=torch.cat((x2,self.up2(d3)),dim=1)
        d2=self.right_block_2(x_cat2)
  
        x_cat1=torch.cat((x1,self.up1(d2)),dim=1)
        d1=self.right_block_1(x_cat1)
        
        output_segmentation_map=self.conv1x1(d1)
        
        return output_segmentation_map
    
    
    
    

# Data transformations for MNIST (resize to U-Net input size)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the U-Net input size
    transforms.ToTensor(),
])



class ISIC2018(torch.utils.data.Dataset):
    def __init__(self, root,transform):
        super().__init__()

        self.img_path = os.path.join(root, 'ISIC2018_Task1-2_Training_Input')
        self.gt_path = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth')

        
        
        self.img_files = sorted(glob.glob(f'{self.img_path}/ISIC_000*.jpg'))
        self.gt_files = sorted(glob.glob(f'{self.gt_path}/ISIC_000*.png'))
        
        self.transform=transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        this_img=Image.open(self.img_files[index])
        this_gt=Image.open(self.gt_files[index])
        
        
        this_img=self.transform(this_img).to(device)
        this_gt=self.transform(this_gt).to(device)
        
        return this_img, this_gt
        

train_loader=torch.utils.data.DataLoader(ISIC2018(root='/home/jian/dataset',transform=transform),batch_size=16)


model=UNet(img_ch=3).to(device)



import torch.optim as optim

# Define the loss function (e.g., cross-entropy)
criterion = nn.MSELoss()

# Define the optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, gts in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, gts)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

torch.save(model, 'model.pth')
