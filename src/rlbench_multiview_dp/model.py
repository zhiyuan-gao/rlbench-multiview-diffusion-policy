import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _group_count(channels: int, max_groups: int = 32) -> int:
    groups = min(int(max_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(_group_count(child.num_features), child.num_features, affine=True)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)
    return module


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, x):
        batch, channels, height, width = x.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            indexing="xy",
        )
        pos_x = pos_x.reshape(1, 1, height * width)
        pos_y = pos_y.reshape(1, 1, height * width)
        flat = x.reshape(batch, channels, height * width)
        weights = F.softmax(flat / max(self.temperature, 1e-6), dim=-1)
        expected_x = torch.sum(weights * pos_x, dim=-1)
        expected_y = torch.sum(weights * pos_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class ResNetConvBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", imagenet_pretrained: bool = False, group_norm: bool = True):
        super().__init__()
        name = str(name).lower()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
            net = models.resnet18(weights=weights)
            self.out_channels = 512
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if imagenet_pretrained else None
            net = models.resnet34(weights=weights)
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone {name!r}; use resnet18 or resnet34")
        if group_norm:
            replace_bn_with_gn(net)
        self.trunk = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

    def forward(self, x):
        return self.trunk(x)


class VisualCore(nn.Module):
    """Robomimic-style visual core: ResNetConv -> SpatialSoftmax -> projection."""

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 64,
        imagenet_pretrained: bool = False,
        group_norm: bool = True,
        spatial_softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.backbone = ResNetConvBackbone(backbone, imagenet_pretrained, group_norm)
        self.pool = SpatialSoftmax(temperature=spatial_softmax_temperature)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone.out_channels * 2),
            nn.Linear(self.backbone.out_channels * 2, int(feature_dim)),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = int(feature_dim)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats)
        return self.proj(pooled)


class MultiViewObsEncoder(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        num_views: int,
        proprio_dim: int,
        text_dim: int,
        visual_backbone: str = "resnet18",
        visual_feature_dim: int = 64,
        imagenet_pretrained: bool = False,
        group_norm: bool = True,
        share_visual_encoder: bool = False,
        global_cond_dim: int = 512,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.num_views = int(num_views)
        self.visual_feature_dim = int(visual_feature_dim)
        self.share_visual_encoder = bool(share_visual_encoder)
        if self.share_visual_encoder:
            self.visual = VisualCore(
                backbone=visual_backbone,
                feature_dim=visual_feature_dim,
                imagenet_pretrained=imagenet_pretrained,
                group_norm=group_norm,
            )
            self.visual_cores = None
        else:
            self.visual = None
            self.visual_cores = nn.ModuleList(
                [
                    VisualCore(
                        backbone=visual_backbone,
                        feature_dim=visual_feature_dim,
                        imagenet_pretrained=imagenet_pretrained,
                        group_norm=group_norm,
                    )
                    for _ in range(self.num_views)
                ]
            )
        self.view_embed = nn.Parameter(torch.zeros(1, 1, self.num_views, self.visual_feature_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.obs_horizon, 1, self.visual_feature_dim))
        flat_visual_dim = self.obs_horizon * self.num_views * self.visual_feature_dim
        flat_proprio_dim = self.obs_horizon * int(proprio_dim)
        fusion_dim = flat_visual_dim + flat_proprio_dim + int(text_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(fusion_hidden_dim)),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(int(fusion_hidden_dim), int(global_cond_dim)),
            nn.Mish(),
        )
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def visual_parameters(self):
        if self.share_visual_encoder:
            return self.visual.parameters()
        return self.visual_cores.parameters()

    def forward(self, images, proprio, text_token):
        # images: [B,T,V,3,H,W], proprio: [B,T,P], text_token: [B,D]
        batch, obs_horizon, num_views = images.shape[:3]
        if obs_horizon != self.obs_horizon or num_views != self.num_views:
            raise ValueError(
                f"Expected images [B,{self.obs_horizon},{self.num_views},3,H,W], got {tuple(images.shape)}"
            )
        if self.share_visual_encoder:
            x = images.reshape(batch * obs_horizon * num_views, *images.shape[3:])
            feats = self.visual(x).reshape(batch, obs_horizon, num_views, self.visual_feature_dim)
        else:
            view_feats = []
            for view_idx, visual in enumerate(self.visual_cores):
                x = images[:, :, view_idx].reshape(batch * obs_horizon, *images.shape[3:])
                view_feats.append(visual(x).reshape(batch, obs_horizon, self.visual_feature_dim))
            feats = torch.stack(view_feats, dim=2)
        feats = feats + self.view_embed + self.time_embed
        flat = torch.cat(
            [
                feats.reshape(batch, -1),
                proprio.reshape(batch, -1),
                text_token.reshape(batch, -1),
            ],
            dim=-1,
        )
        return self.fusion(flat)


def sinusoidal_embedding(timesteps, dim: int):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        groups = _group_count(out_channels, 8)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        y = self.conv1(x)
        y = self.norm1(y)
        scale, bias = self.cond(cond).chunk(2, dim=-1)
        y = y * (1.0 + scale[..., None]) + bias[..., None]
        y = F.mish(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.mish(y)
        return y + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """Compact ConditionalUnet1D-style head for sparse waypoint horizons."""

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = int(global_cond_dim) + int(diffusion_step_embed_dim)
        dims = [int(x) for x in down_dims]
        self.in_proj = nn.Conv1d(self.action_dim, dims[0], 1)
        self.down_blocks = nn.ModuleList()
        prev = dims[0]
        for dim in dims:
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(prev, dim, cond_dim, kernel_size),
                        ConditionalResidualBlock1D(dim, dim, cond_dim, kernel_size),
                    ]
                )
            )
            prev = dim
        self.mid = nn.ModuleList(
            [
                ConditionalResidualBlock1D(dims[-1], dims[-1], cond_dim, kernel_size),
                ConditionalResidualBlock1D(dims[-1], dims[-1], cond_dim, kernel_size),
            ]
        )
        self.up_blocks = nn.ModuleList()
        for skip_dim in reversed(dims):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(prev + skip_dim, skip_dim, cond_dim, kernel_size),
                        ConditionalResidualBlock1D(skip_dim, skip_dim, cond_dim, kernel_size),
                    ]
                )
            )
            prev = skip_dim
        self.out = nn.Sequential(
            nn.Conv1d(dims[0], dims[0], kernel_size, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dims[0], self.action_dim, 1),
        )
        self.diffusion_step_embed_dim = int(diffusion_step_embed_dim)

    def forward(self, noisy_action, timesteps, global_cond):
        # noisy_action: [B,H,A]
        t = self.time_mlp(sinusoidal_embedding(timesteps, self.diffusion_step_embed_dim))
        cond = torch.cat([global_cond, t], dim=-1)
        x = noisy_action.transpose(1, 2)
        x = self.in_proj(x)
        skips = []
        for block1, block2 in self.down_blocks:
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
        for block in self.mid:
            x = block(x, cond)
        for block1, block2 in self.up_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
        return self.out(x).transpose(1, 2)


class MultiViewDiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        num_views: int,
        proprio_dim: int,
        text_dim: int,
        action_dim: int = 7,
        action_horizon: int = 1,
        visual_backbone: str = "resnet18",
        visual_feature_dim: int = 64,
        imagenet_pretrained: bool = False,
        group_norm: bool = True,
        share_visual_encoder: bool = False,
        global_cond_dim: int = 512,
        fusion_hidden_dim: int = 512,
        unet_dims: Sequence[int] = (256, 512, 1024),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.obs_encoder = MultiViewObsEncoder(
            obs_horizon=obs_horizon,
            num_views=num_views,
            proprio_dim=proprio_dim,
            text_dim=text_dim,
            visual_backbone=visual_backbone,
            visual_feature_dim=visual_feature_dim,
            imagenet_pretrained=imagenet_pretrained,
            group_norm=group_norm,
            share_visual_encoder=share_visual_encoder,
            global_cond_dim=global_cond_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        self.noise_pred_net = ConditionalUnet1D(
            action_dim=action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=unet_dims,
        )

    def forward(self, noisy_action, timesteps, images, proprio, text_token):
        global_cond = self.obs_encoder(images, proprio, text_token)
        return self.noise_pred_net(noisy_action, timesteps, global_cond)
