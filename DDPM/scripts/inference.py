import pytorch_lightning as pl

from core.module.DDPM.myDDPM import DDPM
# from core.module.DDPM.DDPM import DDPM
from core.data.CelebaHQ import get_ldmcelebahq

from pytorch_lightning.callbacks import ModelCheckpoint

import yaml
import torch



with open("/home/jian/git_all/codebase/DDPM/config/test.yaml", "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

unet_config=yaml_data['model']['params']['unet_config'] 

dataset_config=yaml_data['data']


# 先不管dataset的东西吧

test=DDPM.load_from_checkpoint('/home/jian/git_all/codebase/DDPM/myDDPMmodel.ckpt',unet_config=unet_config).to('cuda')


img=test.sampling_step()
print(type(img))
print(img)







