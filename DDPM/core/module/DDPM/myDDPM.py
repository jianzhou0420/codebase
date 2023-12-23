# for functions that only use once, I encapsulate them in nested function.
# 有时候你会发现我很多重复的操作，它们是为了增加可读性而添加的。且几乎不会增加计算量，计算时间。
import torch.nn as nn
import torch
import numpy as np
from functools import partial

import pytorch_lightning as pl



def get_beta_schedule(beta_schedule, *, beta_start=0.01, beta_end=10, num_diffusion_timesteps=1000):# 没找到stat和end的默认值
    
    def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        warmup_time = int(num_diffusion_timesteps * warmup_frac)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
        return betas

    def cosine_beta_schedule(timesteps, s = 0.008): # this come from Improved-ddpm
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, a_min = 0, a_max = 0.999)


    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule=='cosine':
        betas=cosine_beta_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    

        
    return betas



class DDPM(pl.LightningModule):
    def __init__(self,unet_config):
        
        
        # 1. define some meta variables 
        self.num_timesteps=1000
        
        
        
        
        # 2.register some coefficients that need to be saved as part of the model, which hence will be used when inferecing.
        self.register_coefficients()
        pass
    
    
    def register_coefficients(self):
        
        # tools
        to_torch=partial(torch.tensor,dtype=torch.float32)
        
        # meata variable
        betas=get_beta_schedule('cosine') # assume it is ndarray
        alphas=1.-betas
        alphas_bar=np.cumprod(alphas,axis=0)
        # alphas_bar_prev=np.append(1.,alphas_bar[:-1]) # 这个是t-1时刻的alphas_bar, 可便于计算，但不便于理解，可以忽略
        
        
        # register meta variable
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alpha_bar', to_torch(alphas_bar))
        self.register_buffer('sqrt_alpha_bar', to_torch(np.sqrt(alphas_bar)))
        self.register_buffer('sqrt_one_minus_alpha_bar', to_torch(np.sqrt(1.-alphas_bar)))

        

        
        pass
    def register_schedule(self):
        pass
    
    
    def training_step(self,batch,batch_idx):
        # pre-process, to make x_0 be [batch_size, width, height, chanel]
        
        # TODO: make it more general
        x0=batch['image'].permute(0,3,1,2)
        # batch : [batch_size,width,height,chanel]
        
        x0=x0
        t= torch.randint(0,self.num_timesteps,(x0.shape[0],),device=self.device)
        noise=torch.randn_like(x0)
        batch_size=x0.shape[0]
        
        x_t=(# equation:√(a_hat) * x_0 + √(1-a_hat) * N(0,I)
            self.sqrt_alphas_bar[t].view(batch_size,1,1,1) * x0+ 
            self.sqrt_one_minus_alphas_cumprod[t].view(batch_size,1,1,1)*noise 
            )
        
        
        model_output=self.model(x_t,t)
        target=noise
        
        # get loss
        loss_dict={}
        loss=torch.nn.functional.mse_loss(target,model_output)
        
        return loss
        
    def sampling_step(self):
        x_t=torch.randn(3,256,256,device=self.device) # random a noisy image x_t
        
        for t in reversed(range(1,self.num_timesteps)): # prograsively denoise
            z=torch.randn(3,256,256,device=self.device)
           
            noise=self.model(x_t,t)

            x_0_predicted=self.math_helper.predict_x_0(self.sqrt_recip_alphas_cumprod[t],x_t,noise)
            
            x_tm1=(self.posterior_mean_coef1[t] *x_0_predicted*self.posterior_mean_coef2[t]*x_t)+self.posterior_variance[t]*z
            
            x_t=x_tm1
        img=x_tm1
        
        return img
        