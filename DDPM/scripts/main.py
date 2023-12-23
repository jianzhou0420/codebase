import pytorch_lightning as pl

from core.module.DDPM.DDPM import DDPM

from pytorch_lightning.callbacks import ModelCheckpoint

from core.data.CelebaHQ import get_ldmcelebahq

import yaml


checkpoint_callback = ModelCheckpoint(
    dirpath='saved_models/',
    filename='model',

    save_top_k=1,  # Save the best model based on validation loss
    mode='min',     # 'min' or 'max' depending on the metric being monitored
    verbose=True
)



# 第一步，先load config并且分给unet和dataset

with open("/home/jian/git_all/codebase/DDPM/config/test.yaml", "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

unet_config=yaml_data['model']['params']['unet_config'] # 生效了！

dataset_config=yaml_data['data']


# 先不管dataset的东西吧
test=DDPM(unet_config)
trainer=pl.Trainer(devices=1,max_epochs=5,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu')
train_loader,val_loader=get_ldmcelebahq(batch_size=3)


trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)



