# this is my own Implementation of DDPM, utlising pytorch lightning.
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from functools import partial

from ..unet.unet import UNetModel



class math_helper:
    def __init__(self):
        # this class is a collection of functions that are used in the DDPM
        pass
    
    def predict_x_0(self,sqrt_recip_alphas_cumprod,x_t,noise):
        
        coeff=sqrt_recip_alphas_cumprod
        output=coeff*(x_t-coeff*noise)
        
        return output


class DDPM(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        
        
        self.model= UNetModel(
        in_channels=3,
        model_channels=224,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8,4,2),
        dropout=0,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,)
        
        
        # lightning configuration
        # logic of this DDPM
        # 1. register_schedule: calculate some parameters
        # 2. tran_step
        self.parameterization = "eps"
        self.l_simple_weight=1.
        self.original_elbo_weight=0.
        self.math_helper=math_helper()
        self.register_schedule(timesteps=1000)# 计算一些类似于alpha，beta的参数

    def register_schedule(self,v_posterior=0, beta_schedule="linear", timesteps=1000, 
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        # v_posterior是improved DDPM的一部分，先默认为0。它类似一个权重

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas)) # 知道
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))# 知道
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))# 约等于t-1

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))# 知道
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))# 知道
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod))) # 知道
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod))) # TODO： 这是什么
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1))) # TODO： 这是什么

        # calculations for posterior q(x_{t-1} | x_t, x_0) # 这里是插值那部分，算x_{t-1}的
        posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
        
        
        
        
        # ldm作者的forward写得及其不合理，算loss的过程其实是training_step的过程，而非
        # 不如就不要forward 部分了，反正training_step里面才是真正的forward。而且diffusion也没有传统意义上的forward部分。
        # diffusion的forward，可以说是加噪声的过程。
        # 不要forward了！
    def forward(self,x_t,t):
        return self.model(x_t,t)
    
    

    
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        # batch : [batch_size,width,height,chanel]
        x0=batch['image'].permute(0,3,1,2)
        t= torch.randint(0,self.num_timesteps,(x0.shape[0],),device=self.device)
        noise=torch.randn_like(x0)
        
        batch_size=x0.shape[0]
   
        x_t=self.sqrt_alphas_cumprod[t].view(batch_size,1,1,1) * x0+self.sqrt_one_minus_alphas_cumprod[t].view(batch_size,1,1,1)*noise
        
        model_output=self.model(x_t,t)
        
        target=noise
        
        # get loss
        loss_dict={}
        loss=torch.nn.functional.mse_loss(target,model_output)
        
        
        ## TODO： 以下好像是Improved DDPM的内容
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})
        
        return loss

    def validation_step(self,val_batch,batch_ix):
        pass
    
    # lightningModel hooks, lightningmodules has +20 hooks to keep all the flexibility
    def p_sample(self):
        x_t=torch.randn(3,256,256,device=self.device)
        
        for t in reversed(range(1,self.num_timesteps)):
            z=torch.randn(3,256,256,device=self.device)
           
            noise=self.model(x_t,t)

            x_0_predicted=self.math_helper.predict_x_0(self.sqrt_recip_alphas_cumprod[t],x_t,noise)
            
            x_tm1=(self.posterior_mean_coef1[t] *x_0_predicted*self.posterior_mean_coef2[t]*x_t)+self.posterior_variance[t]*z
            
            x_t=x_tm1
        img=x_tm1
        
        return img
        


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

