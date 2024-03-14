import pytorch_lightning as pl
from core.module.DDPM.myDDPM import GaussianDiffusion

from core.data.cifar10 import get_cifar10

from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import matplotlib.pyplot as plt
import numpy as np


class DDPM_trainer(pl.LightningModule,GaussianDiffusion):
    def __init__(self):
        super().__init__()
        GaussianDiffusion.__init__(self,timesteps=1000,device=torch.device('cuda'),precision=torch.float32)
        # torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision('medium')
        
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=2e-6)
        return optimizer
    
    def training_step(self,batch):
        x_0=batch[0].to(self.precision)# for cifar
        t= torch.randint(0,self.timesteps,(x_0.shape[0],),device=self.device)
        loss=self.training_losses(x_0,t)
        self.log('loss',loss,prog_bar=True,logger=True,on_epoch=False)
        return loss
 


def npshow(x_0,save):
    
    image_np = x_0.cpu().numpy().squeeze()
    batch_size,_,_,_=image_np.shape
    
    
    print('max=',np.max(image_np),'\nmin',np.min(image_np),'\nmean',np.mean(image_np))

    # Transpose the array to have the channels as the last dimension
    for i in range(batch_size):
        item=image_np[i]
        item = np.transpose(item, (1, 2, 0))


        item=(item-np.min(item))/(np.max(item)-np.min(item))
        # item=np.absolute(item)/np.max(item)
    # Display the image using Matplotlib
        if save:
            from datetime import datetime
            current_time = str(datetime.now())
            plt.imsave(current_time+'.png',item)

def train_cifar():
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename='model',
        save_top_k=1,  # Save the best model based on validation loss
        mode='min',     # 'min' or 'max' depending on the metric being monitored
        verbose=True
    )


    train_loader,val_loader=get_cifar10(23)


    trainer=pl.Trainer(devices=1,max_epochs=20,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu')
    test=DDPM_trainer().to('cuda')
    trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)


def sample_cifar_from_ckpt(model_path,save,batch_size,num_batch,image_size=32):
    test=DDPM_trainer.load_from_checkpoint(model_path).to('cuda')
    shape=[batch_size,3,image_size,image_size]
    for i in range(num_batch):
        img=test.p_sample_loop(shape)
        npshow(img,save)
    


def train_cifar_from_ckpt(model_path):
    '''
    continue to train a model
    '''
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename='model',
        save_top_k=1,  # Save the best model based on validation loss
        mode='min',     # 'min' or 'max' depending on the metric being monitored
        verbose=True
    )


    train_loader,val_loader=get_cifar10(15)


    trainer=pl.Trainer(devices=1,max_epochs=20,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu')
    test=DDPM_trainer.load_from_checkpoint(model_path).to(torch.device('cuda'))
    trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)




if __name__=='__main__':
    # train_cifar_from_ckpt('/home/jian/git_all/codebase/DDPM/saved_models/model-v3.ckpt')

    sample_cifar_from_ckpt('/home/jian/git_all/codebase/DDPM/saved_models/model-v4.ckpt',save=True,batch_size=1,num_batch=1,image_size=256)
    # train_cifar()