# for functions that only use once, I encapsulate them in nested function.
# 有时候你会发现我很多重复的操作，它们是为了增加可读性而添加的。且几乎不会增加计算量，计算时间。

import torch.nn as nn
import torch
import numpy as np
from functools import partial

import pytorch_lightning as pl
from core.module.unet.unet import UNetModel
# from core.module.unet.openaimodel import UNetModel # 也会溢出
from tqdm import tqdm




# test memory profiler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# /test
device='cuda'





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
    def __init__(self):
        super().__init__()
        
        
        # 1. define some meta variables 
        self.num_timesteps=1000
        
        
        # 2.register some coefficients that need to be saved as part of the model, which hence will be used when inferecing.
        self.register_coefficients()
        
        # 3.define model
        self.model= UNetModel(
        # image_size=64,
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

        # 4. 暂存
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer
    
    
    def register_coefficients(self):
        
        # tools
        to_torch=partial(torch.tensor,dtype=torch.float32)
        
        # meata variable
        betas=get_beta_schedule('cosine',num_diffusion_timesteps=self.num_timesteps) # assume it is ndarray
        alphas=1.-betas
        
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        # to_torch
        
        self.alphas_cumprod=to_torch(self.alphas_cumprod).to(device)
        self.alphas_cumprod_prev=to_torch(self.alphas_cumprod_prev).to(device)
        self.alphas_cumprod_next=to_torch(self.alphas_cumprod_next).to(device)
        self.sqrt_alphas_cumprod=to_torch(self.sqrt_alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod=to_torch(self.sqrt_one_minus_alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod=to_torch(self.log_one_minus_alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod=to_torch(self.sqrt_recip_alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod=to_torch(self.sqrt_recipm1_alphas_cumprod).to(device)
        self.posterior_variance=to_torch(self.posterior_variance).to(device)
        self.posterior_log_variance_clipped=to_torch(self.posterior_log_variance_clipped).to(device)
        self.posterior_mean_coef1=to_torch(self.posterior_mean_coef1).to(device)

    
    def register_schedule(self):
        pass
    
    
    def training_step(self,batch,batch_idx):
        # pre-process, to make x_0 be [batch_size, width, height, chanel]
        
        # 
        x0=batch['image'].permute(0,3,1,2)
        # batch : [batch_size,width,height,chanel]
        t= torch.randint(0,self.num_timesteps,(x0.shape[0],),device=self.device)
        noise=torch.randn_like(x0)
        batch_size=x0.shape[0]
        
        # 逻辑是，
        # 先用原图x_0,通过不同的t，得到经过t次加噪音noise之后的x_t
        # 然后，model，告诉model，x_t和t，让它推理噪音是什么样子。
        # 因此，model的输出是，噪音
        x_t=(# equation:√(a_hat) * x_0 + √(1-a_hat) * N(0,I)
            self.sqrt_alphas_cumprod[t].view(batch_size,1,1,1) * x0+ 
            self.sqrt_one_minus_alphas_cumprod[t].view(batch_size,1,1,1)*noise 
            )
        
        
        model_output=self.model(x_t,t)
        target=noise
        
        # get loss
        loss=torch.nn.functional.mse_loss(target,model_output)
        print(model_output.max(),model_output.min())
        # log
        self.log('loss',loss,prog_bar=True,logger=True,on_epoch=False)
        return loss
    
    
    @torch.no_grad()
    def sampling_step(self):
        self.model.eval()
        x_t=torch.randn(1,3,64,64,device=device) # random a noisy image x_t # TODO: 1，64，都可以变成参数
        
        for t in tqdm(reversed(range(1,self.num_timesteps))): # prograsively denoise
            
            t=torch.tensor(t,device=device).reshape(1)
            z=torch.randn(3,64,64,device=device)

            if t==1000-24:
                print(1)
                pass


            noise=self.model(x_t,t)
            
            if t==1000-24:
                print(1)
                pass
            
            x_0_predicted=(x_t-self.sqrt_one_minus_alphas_cumprod[t]*noise)/self.sqrt_alphas_cumprod[t]
            
            x_tm1=(self.posterior_mean_coef1 *x_0_predicted+self.posterior_mean_coef2*x_t)+self.posterior_variance*z
            
            x_t=x_tm1

        img=x_t # at present, x_t is the x_0_predicted
        
        #test
        # t=torch.tensor(1,device='cuda').reshape(1)
        # noise=self.model(x_t,t)
        # img=noise
        #/test
        
        return img
        