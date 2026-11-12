

import numpy as np 
import torch 
from tqdm import tqdm 

class DDIM:
    def __init__(self) -> None:
        self.T = 1000
        self.alpha = torch.sqrt(1 - 0.02 * torch.arange(1, self.T + 1) / self.T)
        self.beta = torch.sqrt(1 - self.alpha**2)
        self.bar_alpha = torch.cumprod(self.alpha, dim=0)
        self.bar_beta = torch.sqrt(1 - self.bar_alpha**2)
        self.sigma = self.beta.clone()

    def q_sample(self, x_start):
        device = x_start.device
        batch_size = len(x_start)
        batch_steps = torch.randint(low=0, high=self.T, size=(batch_size,))
        batch_bar_alpha = self.bar_alpha[batch_steps][:, None, None, None].to(device)
        batch_bar_beta = self.bar_beta[batch_steps][:, None, None, None].to(device)
        batch_noise = torch.randn(*x_start.shape).to(device)
        batch_steps = batch_steps.to(device)
        batch_noisy_imgs = x_start * batch_bar_alpha + batch_noise * batch_bar_beta

        return batch_noisy_imgs, batch_steps, batch_noise

    
    def sample(self, model, shape):
        device = next(model.parameters()).device
        n, channel, img_size, img_size = shape
        z_samples = torch.randn(n**2, channel, img_size, img_size).to(device)
       
        for t in tqdm(range(self.T), total=self.T):
            t = self.T - t - 1
            bt = torch.tensor([t] * z_samples.shape[0]).to(device)
            beta_t = (self.beta[t]**2).to(device)
            bar_beta_t = self.bar_beta[t].to(device)
            alpha_t = self.alpha[t].to(device)
            sigma_t = self.sigma[t].to(device)
            with torch.no_grad():
                z_samples -= beta_t / bar_beta_t * model(z_samples, bt)

            z_samples /= alpha_t
            ne = torch.randn(*z_samples.shape).to(device)
            z_samples += ne * sigma_t

        x_samples = torch.clamp(z_samples, -1, 1)

        return x_samples

    def ddim_sample(self, model, shape, stride=1, eta=1, **kwargs):
        device = next(model.parameters()).device
        # 采样参数
        bar_alpha_ = self.bar_alpha[::stride]
        bar_alpha_pre_ = torch.nn.functional.pad(bar_alpha_[:-1], [1, 0], mode="constant", value=1)
        # bar_alpha_pre_ = torch.nn.functional.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
        bar_beta_ = torch.sqrt(1 - bar_alpha_**2)
        bar_beta_pre_ = torch.sqrt(1 - bar_alpha_pre_**2)
        alpha_ = bar_alpha_ / bar_alpha_pre_
        sigma_ = bar_beta_pre_ / bar_beta_ * torch.sqrt(1 - alpha_**2) * eta
        epsilon_ = bar_beta_ - alpha_ * torch.sqrt(bar_beta_pre_**2 - sigma_**2)
        T_ = len(bar_alpha_)
        # 采样过程
        n, channel, img_size, img_size = shape
        z_samples = torch.randn(n, channel, img_size, img_size).to(device)
       
        for t in tqdm(range(T_), ncols=0):
            t = T_ - t - 1
            bt = torch.tensor([t * stride] * z_samples.shape[0]).to(device)
            epsilon_t = epsilon_[t].to(device)
            alpha_t = alpha_[t].to(device)
            sigma_t = sigma_[t].to(device)
            with torch.no_grad():
                z_samples -= epsilon_t * model(z_samples, bt, **kwargs)
            z_samples /= alpha_t
            ne =  torch.randn(*z_samples.shape).to(device)
            z_samples += ne * sigma_t

        x_samples = torch.clamp(z_samples, -1, 1)
        return x_samples