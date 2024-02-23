import pytorch_lightning as pl
# from core.module.DDPM.DDPM import DDPM
from core.module.DDPM.myDDPM import DDPM

from core.data.CelebaHQ import get_ldmcelebahq

from pytorch_lightning.callbacks import ModelCheckpoint

import yaml


checkpoint_callback = ModelCheckpoint(
    dirpath='saved_models/',
    filename='model',
    save_top_k=1,  # Save the best model based on validation loss
    mode='min',     # 'min' or 'max' depending on the metric being monitored
    verbose=True
)



train_loader,val_loader=get_ldmcelebahq(dataroot="/home/jian/git_all/datasets/celebahq",
                                        listroot="/home/jian/git_all/codebase/DDPM/core/data",batch_size=15)
# 重新训练
trainer=pl.Trainer(devices=1,max_epochs=5,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu')
test=DDPM()
trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)

# 接着训练
# test=DDPM()
# trainer=pl.Trainer(devices=1,max_epochs=5,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu')
# trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path='/home/jian/git_all/codebase/DDPM/2epochs.ckpt')



