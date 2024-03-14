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
        
        
        # flow control
        self.loss_list=[]
        
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=2e-6)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        # flow control
        self.flow_control_before(batch_idx)
        # training step
        x_0=batch[0].to(self.precision)# for cifar
        t= torch.randint(0,self.timesteps,(x_0.shape[0],),device=self.device)
        loss=self.training_losses(x_0,t)
        loss=loss.mean()
        self.log('loss',loss,prog_bar=True,logger=True,on_epoch=False)
        self.loss_list.append(loss.item())
        self.flow_control_behind(batch_idx)
        return loss
 
    def flow_control_before(self,batch_idx):
        pass
        # Flow control 1: stop when loss is smaller than a threshold
        # loss_threshold=0.8
        # if batch_idx==0 and self.current_epoch!=0:
        #     mean=np.mean(self.loss_list)
        #     if mean<=loss_threshold:
        #         self.trainer.should_stop=True
        #         print('stopped when loss mean is smaller than',loss_threshold,'\n current loss mean=',mean)
        #     self.loss_list=[]
        # assert len(self.loss_list)==batch_idx
        
        

    def flow_control_behind(self,batch_idx):
        # Flow control 2: For every two epoch, Record average loss, Sample 10 times, and save the model.
        if batch_idx==0 and self.current_epoch%2==0 and self.current_epoch!=0: #每两个epoch记录一次
            # save
            self.trainer.save_checkpoint(filepath='saved_models/test'+str(self.current_epoch)+'.ckpt')
            # sample
            num_batch=1
            batch_size=10
            shape=[batch_size,3,32,32]
            for i in range(num_batch):
                img=self.p_sample_loop(shape)
                npshow(img,save=True,path='saved_models/sample'+str(self.current_epoch)+'.png')
        pass

def npshow(x_0,save,path=None):
    image_np = x_0.cpu().numpy()
    batch_size,*_ =image_np.shape
    
    
    print('max=',np.max(image_np),'\nmin',np.min(image_np),'\nmean',np.mean(image_np))

    # Transpose the array to have the channels as the last dimension
    for i in range(batch_size):
        item=image_np[i]
        item = np.transpose(item, (1, 2, 0))
        # item=(item-np.min(item))/(np.max(item)-np.min(item))
        # item=np.absolute(item)/np.max(item)
        item=(item-(-1))/2
    # Display the image using Matplotlib
        if save:
            from datetime import datetime
            current_time = str(datetime.now())
            
            if path!=None:
                plt.imsave(path,item)
            else:
                plt.imsave(current_time+'.png',item)

def train_cifar(batch_size,epoch):
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename='model',
        save_top_k=1,  # Save the best model based on validation loss
        mode='min',     # 'min' or 'max' depending on the metric being monitored
        verbose=True
    )


    train_loader,val_loader=get_cifar10(batch_size)


    trainer=pl.Trainer(devices=1,max_epochs=epoch,callbacks=[checkpoint_callback],limit_val_batches=10,accelerator='gpu',log_every_n_steps=1)
    test=DDPM_trainer().to('cuda')
    trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)


def sample_cifar_from_ckpt(model_path,save,batch_size,num_batch,image_size=32):
    test=DDPM_trainer.load_from_checkpoint(model_path).to('cuda')
    shape=[batch_size,3,image_size,image_size]
    for i in range(num_batch):
        img=test.p_sample_loop(shape)
        npshow(img,save)
    


def train_cifar_from_ckpt(model_path,batch_size,epoch):
    '''
    continue to train a model
    '''


    train_loader,val_loader=get_cifar10(batch_size)


    trainer=pl.Trainer(devices=1,max_epochs=epoch,limit_val_batches=10,accelerator='gpu')
    test=DDPM_trainer.load_from_checkpoint(model_path).to(torch.device('cuda'))
    trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader)


def continue_to_train(batch_size,max_epoch,path):
    
    train_loader,val_loader=get_cifar10(batch_size)
    test=DDPM_trainer().to('cuda')
    trainer=pl.Trainer(devices=1,max_epochs=max_epoch,limit_val_batches=10,accelerator='gpu')
    trainer.fit(test,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=path)


