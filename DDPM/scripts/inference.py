import pytorch_lightning as pl

from z.myDDPM import DDPM
# from core.module.DDPM.DDPM import DDPM
from core.data.CelebaHQ import get_ldmcelebahq

from pytorch_lightning.callbacks import ModelCheckpoint

import yaml
import torch



# with open("/home/jian/git_all/codebase/DDPM/config/test.yaml", "r") as file:
#     yaml_data = yaml.load(file, Loader=yaml.FullLoader)

# unet_config=yaml_data['model']['params']['unet_config'] 

# dataset_config=yaml_data['data']


# 先不管dataset的东西吧

test=DDPM.load_from_checkpoint('/home/jian/git_all/codebase/DDPM/model-v1.ckpt').to('cuda')

shape=[1,3,64,64]
img=test.sampling_step()

import matplotlib.pyplot as plt
import numpy as np

# Assuming your tensor is named 'image_tensor'
# Convert the tensor to a NumPy array
image_np = img.cpu().numpy().squeeze()

# Transpose the array to have the channels as the last dimension
image_np = np.transpose(image_np, (1, 2, 0))


print('max=',np.max(image_np),'\nmin',np.min(image_np),'\nmean',np.mean(image_np))
image_np=(image_np-np.min(image_np))/(np.max(image_np)-np.min(image_np))
plt.imsave('t.png',image_np)





