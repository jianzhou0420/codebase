
import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import pytorch_lightning as pl
from core.module.unet.unet import UNetModel
# from core.module.unet.openaimodel import UNetModel # 也会溢出
from tqdm import tqdm
import math

# gaussian diffusion trainer class
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    out=out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out

class GaussianDiffusion(nn.Module):
    def __init__(self,
                 timesteps=1000,
                 beta_schedule='cosine',
                 device=torch.device('cuda'),
                 precision=torch.float64):
        super().__init__()
        
        # TODO: precision
        self.mydevice=device
        self.timesteps=int(timesteps)
        self.precision=precision
        
        # 1. Define neural network model
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
                    use_scale_shift_norm=False,
                    )
        
        # 2. Define some parameters
        
        # 3. Define meta coefficients
        
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        betas=beta_schedule_fn(timesteps) # this is torch.tensor
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # helper function to register buffer from float64 to float32

        rb = lambda name, val: self.register_buffer(name, val.to(self.precision)) # 这一步太天才了

        rb('betas', betas)
        rb('alphas_cumprod', alphas_cumprod)
        rb('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        rb('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        rb('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        rb('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        rb('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        rb('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        rb('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        rb('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        rb('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        rb('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal


    # === Training ===
    def training_losses(self,x_0,t):
        '''
        accepting gt x_0 and the specific timesteps
        
        x_0: [batch_size,3,H,W]
        t: int
        
        1. sample x_t from x_0 (GT)
        2. get model output
        3. calculate loss
        
        return: loss
        '''
        # TODO: set more target type, for now, use noise
        # TODO: set different loss type, for now, use mse
        
        # 0.Get noise
        noise=torch.randn_like(x_0)
        
        # 1.Sample x_t from x_0
        x_t=self.q_sample(x_0,t,noise)
        
        # 2.Get model output
        model_out=self.model(x_t,t)
        
        # 3.Calculate loss
        target=noise
        loss=F.mse_loss(model_out,target,reduction='none')
        loss=loss.mean(dim=[1,2,3]) # 有点问题
        assert loss.shape==t.shape,'loss is not in shape of [batch_size]'
        return loss
    
    def q_sample(self,x_0,t,noise):
        '''
        Diffuse the data from x_0 to x_t
        
        '''
        x_t=(extract(self.sqrt_alphas_cumprod,t,x_0.shape)*x_0+
             extract(self.sqrt_one_minus_alphas_cumprod,t,noise.shape)*noise) # broadcast
        
        return x_t
        
    # === Sampling ===
    @torch.no_grad()
    def p_sample_loop(self,shape):
        '''
        Generate samples: infer x_0 from x_t (random noise).
        
        batch_size: int
        shape: [B,C,H,W],bach_size,channel, height, width
        
        '''
        B,C,H,W=shape
        
        img=torch.randn(shape,device='cuda')
        
        for t in tqdm(reversed(range(0,self.timesteps))):
            t=torch.tensor(t,device='cuda').reshape([1])
            img=self.p_sample(img,t)
            
        return img

    @torch.no_grad()
    def p_sample(self,x_t,t):
        '''
        Sample from the model: infer x_{t-1} from x_t
        
        How?
        Assume model predict epsilon(noise)
        
        Firstly, get x_0 from x_t=√(a_hat) * x_0 + √(1-a_hat) * N(0,I), where N(0,I) is the noise, is the model's output.
        Secondly, calculate the posterior mean and variance: q(x_{t-1}|x_t,x_0)
        
        '''
        # 0. get epsilon(noise)
        model_output=self.model(x_t,t)
        
        # 1. get x_0
        x_0_predicted=(extract(self.sqrt_recip_alphas_cumprod,t,x_t.shape)*x_t-
                       extract(self.sqrt_recipm1_alphas_cumprod,t,model_output.shape)*model_output)
        x_0_predicted=torch.clamp(x_0_predicted,min=-1.,max=1.)
        
        # 2. calculate the posterior mean and variance of q(x_{t-1}|x_t,x_0) lucidrain和原作者，使用太多函数化的东西了，对初学者很不方便。
        model_mean,_,model_log_variance_clipped=self.p_mean_variance(x_0=x_0_predicted,x_t=x_t,t=t)
        
        # 3. get x_{tm1}
        new_noise=torch.randn_like(x_t) # this is different from model_output, be carefule, refer orignal paper for why or refer Hongyi Li's lecture
        
        x_tm1=model_mean+(0.5*model_log_variance_clipped).exp()*new_noise
        
        return x_tm1
    
    
    @torch.no_grad()
    def p_mean_variance(self,x_0,x_t,t):
        posterior_mean=(extract(self.posterior_mean_coef1,t,x_0.shape)*x_0+
                        extract(self.posterior_mean_coef2,t,x_t.shape)*x_t)
        posterior_variance=extract(self.posterior_variance,t,x_t.shape)
        posterior_log_variance_clipped=extract(self.posterior_log_variance_clipped,t,x_t.shape)
        return posterior_mean,posterior_variance,posterior_log_variance_clipped
        
        
        





