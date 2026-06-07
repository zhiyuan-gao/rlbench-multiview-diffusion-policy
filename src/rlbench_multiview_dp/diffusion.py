import copy
from typing import Iterable, List

import numpy as np
import torch


class DDPMNoiseScheduler:
    def __init__(self, train_steps: int = 100, device=None):
        self.train_steps = int(train_steps)
        beta = torch.linspace(1e-4, 2e-2, self.train_steps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def add_noise(self, x0, noise, timesteps):
        shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        a = self.sqrt_alpha_bar[timesteps].reshape(shape)
        b = self.sqrt_one_minus_alpha_bar[timesteps].reshape(shape)
        return a * x0 + b * noise


def ddim_timesteps(train_steps: int, sample_steps: int) -> List[int]:
    train_steps = int(train_steps)
    sample_steps = int(sample_steps)
    if sample_steps >= train_steps:
        steps = np.arange(train_steps, dtype=np.int64)
    else:
        steps = np.linspace(0, train_steps - 1, sample_steps, dtype=np.int64)
        steps = np.unique(steps)
    return list(reversed(steps.tolist()))


@torch.inference_mode()
def sample_ddim(
    model,
    images,
    proprio,
    text_token,
    action_shape,
    action_mean,
    action_std,
    train_steps: int = 100,
    sample_steps: int = 20,
    amp: bool = True,
    amp_dtype=torch.bfloat16,
):
    device = images.device
    scheduler = DDPMNoiseScheduler(train_steps, device=device)
    x = torch.randn(action_shape, device=device)
    mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device).view(1, 1, -1)
    steps = ddim_timesteps(train_steps, sample_steps)
    for i, t in enumerate(steps):
        t_batch = torch.full((action_shape[0],), int(t), device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            eps = model(x, t_batch, images, proprio, text_token)
        a_t = scheduler.alpha_bar[t].view(1, 1, 1)
        pred_x0 = (x - torch.sqrt(1.0 - a_t) * eps.float()) / torch.sqrt(a_t).clamp_min(1e-12)
        if i + 1 < len(steps):
            prev_t = steps[i + 1]
            a_prev = scheduler.alpha_bar[prev_t].view(1, 1, 1)
        else:
            a_prev = torch.ones((1, 1, 1), device=device)
        x = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1.0 - a_prev) * eps.float()
    pred = x * std + mean
    pred[..., -1] = (pred[..., -1] >= 0.5).to(pred.dtype)
    return pred


class EMAModel:
    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self):
        return self.ema_model.state_dict()

    def to(self, device):
        self.ema_model.to(device)
        return self
