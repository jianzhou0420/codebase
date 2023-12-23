import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, OptimizerLRScheduler
import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class litAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,28*28)
        )


    def forward(self,x):
        embedding=self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        x,y=train_batch
       
        x=x.view(x.size(0),-1)
        z=self.encoder(x)
        x_hat=self.decoder(z)
        loss=F.mse_loss(x_hat,x)
        self.log('train_loss',loss,on_epoch=True)
        return loss

    def validation_step(self,val_batch,batch_ix):
        x,y=val_batch
    
        x=x.view(x.size(0),-1)
        z=self.encoder(x)
        x_hat=self.decoder(z)
        loss=F.mse_loss(x_hat,x)
        self.log('val_loss',loss,on_epoch=True)
        return loss
    
    # lightningModel hooks, lightningmodules has +20 hooks to keep all the flexibility
   
# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



model=litAutoEncoder()

trainer=pl.Trainer(limit_train_batches=100,max_epochs=1,devices='auto')
trainer.fit(model,trainloader,testloader) 