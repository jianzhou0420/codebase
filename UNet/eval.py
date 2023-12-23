import torch
import glob
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


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
    
    
transform = transforms.Compose([
transforms.Resize((256, 256)),  # Resize to the U-Net input size
transforms.ToTensor(),
])
import random
testid=random.randint(0,1000)
print(testid)
model=torch.load('model.pth')
image=sorted(glob.glob('/home/jian/git_all/codebase/UNet/ISIC2018_Task1-2_Training_Input/ISIC_*.jpg'))[testid]
gt=sorted(glob.glob('/home/jian/git_all/codebase/UNet/ISIC2018_Task1_Training_GroundTruth/ISIC_*.png'))[testid]

image=Image.open(image)
gt=Image.open(gt)

image=transform(image).to('cuda')
gt=transform(gt)

image=torch.reshape(image,(1,3,256,256))

predict=model(image)

plt.subplot(1,3,1)
image=torch.squeeze(image)
image=image.permute(1,2,0)
plt.imshow(image.cpu().detach().numpy())
plt.title('input')
plt.subplot(1,3,2)
plt.imshow(gt[0],cmap='gray')
plt.title('gt')
plt.subplot(1,3,3)
plt.imshow(predict[0][0].cpu().detach().numpy(),cmap='gray')
plt.title('predicted')

plt.show()

