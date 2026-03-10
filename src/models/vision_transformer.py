# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found at https://github.com/facebookresearch/ijepa/blob/main/LICENSE
#
# Modifications and additional code:
# Copyright (c) 2026 Forschungszentrum Jülich GmbH
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.transforms import get_transformed_grid
from src.masks.utils import apply_masks
from functools import lru_cache

@lru_cache(maxsize=3)
def get_2d_sincos_pos_embed_from_params(embed_dim, H, W, cls_token=False, tx=torch.tensor([[0.]]), ty=torch.tensor([[0.]]), angle=torch.tensor([[0.]]), scale=torch.tensor([[1.0]]), patch_size=16, **kwargs):
    grid = get_transformed_grid(H, W, tx, ty, angle, scale, patch_size) # (B, H, W, 2)
    grid = grid.permute(0, 3, 1, 2)  # (B, 2, H, W)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid) # (B, H*W, D)
    if cls_token:
        pos_embed = torch.cat([torch.zeros((pos_embed.size(0), 1, embed_dim), device=pos_embed.device, dtype=pos_embed.dtype), pos_embed], dim=1)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([1, 2, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, 1, embed_dim]), pos_embed], dim=1)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    if not torch.is_tensor(grid):
        grid = torch.from_numpy(grid)

    assert grid.shape[1] == 2, "Grid must have shape (B, 2, H, W) or (2, H, W)"

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,0])  # (B, H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,1])  # (B, H*W, D/2)

    emb = torch.concatenate([emb_h, emb_w], axis=-1)  # (B, H*W, D)

    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, 1, embed_dim]), pos_embed], axis=1)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(pos.shape[0],-1)   # (B,M,)
    out = torch.einsum('bm,d->bmd', pos, omega)   # (B,M, D/2), outer product

    emb_sin = torch.sin(out)  # (B, M, D/2)
    emb_cos = torch.cos(out)  # (B, M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=-1)  # (B,M, D)
    return emb



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ViT_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ViT_MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)

class MLP(nn.Module):
    """A simple MLP with configurable number of layers and ReLU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, n_layers=2, act_layer=nn.GELU, drop=0.):
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be at least 2, got {n_layers}")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        layers = [
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop)
        ]
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                act_layer(),
                nn.Dropout(drop)
            ])
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Dropout(drop))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        num_crop_patches,
        patch_size=16,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_conditionings=4,
        interp_pos_encoding='conditional',  # 'interpolate' or 'conditional'
        enc_has_cls_token=False,
        n_register_tokens=0,
        no_condition_mlp=False,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        if not no_condition_mlp > 0:
            self.condition_mlp = MLP(predictor_embed_dim+num_conditionings, predictor_embed_dim, predictor_embed_dim, n_layers=3)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches+int(enc_has_cls_token), predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=enc_has_cls_token)
        self.predictor_pos_embed.data.copy_(predictor_pos_embed.float())
        self.interp_pos_encoding= interp_pos_encoding
        self.num_patches = num_patches
        self.num_crop_patches = num_crop_patches
        self.patch_size = patch_size
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.enc_has_cls_token = enc_has_cls_token
        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, predictor_embed_dim))
        self.n_register_tokens = n_register_tokens
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if self.n_register_tokens > 0:
            nn.init.normal_(self.register_tokens, std=self.init_std)

    def forward(self, x, masks_x, masks, conditions={}):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1) #* These should stay constant as original image shape does not change
        if self.enc_has_cls_token:
            cls_pos_embed = x_pos_embed[:, 0].unsqueeze(1)  # (B, 1, D)
            x_pos_embed = x_pos_embed[:, 1:] # (B, N, D)
        x_pos_embed = apply_masks(x_pos_embed, masks_x)
        if self.enc_has_cls_token:
            cls_pos_embed = cls_pos_embed.repeat(len(masks_x), 1, 1)  # (len(masks)*B, 1, D)
            x_pos_embed = torch.cat([cls_pos_embed, x_pos_embed], dim=1)
        x += x_pos_embed

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        if self.interp_pos_encoding == 'interpolate':
            pos_embs = interpolate_pos_encoding(torch.zeros(B,self.num_patches,D), self.predictor_pos_embed, cls_token=self.enc_has_cls_token, x_has_cls_token=False)
            pos_embs = pos_embs.repeat(B, 1, 1)
        elif self.interp_pos_encoding == 'conditional':
            pos_embs = get_2d_sincos_pos_embed_from_params(
                self.predictor_pos_embed.shape[-1],
                int(self.num_crop_patches**.5),
                int(self.num_crop_patches**.5),
                **conditions, # conditions should be a dict with keys: tx, ty, theta_deg, scale
                patch_size=self.patch_size,
                cls_token=self.enc_has_cls_token
            ).to(device=x.device, dtype=x.dtype)
            if pos_embs.size(0) == 1: # should only happen if conditions are empty, otherwise they provide the batchsize
                pos_embs = pos_embs.repeat(B, 1, 1)  # (B, N_mask, D)

        if self.enc_has_cls_token: # remove cls pos embedding, only needed for context tokens, predictor does not have cls token
            pos_embs = pos_embs[:, 1:]
        pos_embs = apply_masks(pos_embs, masks) # (len(masks)*B, n_masked, D)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) # 
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        if len(conditions) > 0:
            # -- concat conditions to pred_tokens
            conditions = torch.cat(list(conditions.values()), dim=1)
            conditions = conditions.unsqueeze(1).repeat(len(masks), pred_tokens.size(dim=1), 1)  # (B, N_mask, num_conditionings)
            conditions = repeat_interleave_batch(conditions, B, repeat=len(masks_x))
            pred_tokens = torch.cat([pred_tokens, conditions], dim=-1)  # (B, N_mask, D + num_conditionings)
            pred_tokens = self.condition_mlp(pred_tokens)  # (B, N_mask, D)

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        if self.n_register_tokens > 0:
            register_tokens = self.register_tokens.expand(B*len(masks), -1, -1)
            x = torch.cat((x, register_tokens), dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        if self.n_register_tokens > 0:
            x = x[:, N_ctxt:-self.n_register_tokens]
        else:
            x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[256],
        crop_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interp_pos_encoding='conditional',  # 'interpolate' or 'conditional'
        cls_token=False,
        n_register_tokens=0,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+int(cls_token), embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=cls_token)
        self.pos_embed.data.copy_(pos_embed.float())
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.has_cls_token = cls_token
        if n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, embed_dim))
        self.n_register_tokens = n_register_tokens
        self.interp_pos_encoding = interp_pos_encoding
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if self.has_cls_token:
            nn.init.normal(self.cls_token, std=self.init_std)
        if self.n_register_tokens > 0:
            nn.init.normal_(self.register_tokens, std=self.init_std)

    def forward(self, x, masks=None):
        #* the encoder should never use the conditioning parameters
        #* always act as if it were the original image (but with differing image sizes)

        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        if self.interp_pos_encoding == 'interpolate':
            pos_embed = interpolate_pos_encoding(x, self.pos_embed, cls_token=self.has_cls_token, x_has_cls_token=False)
        elif self.interp_pos_encoding == 'conditional':
            pos_embed = get_2d_sincos_pos_embed_from_params(
                embed_dim=self.pos_embed.shape[-1],
                H=int(N**.5),
                W=int(N**.5),
                patch_size=self.patch_embed.patch_size,
                cls_token=self.has_cls_token,
            )
            pos_embed = pos_embed.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown interpolate_pos_encoding: {self.interp_pos_encoding}")

        if self.has_cls_token:
            pos_embed = pos_embed[:, 1:]  # remove cls token from pos_embed

        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- add cls token
        if self.has_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1) + pos_embed[:, :1]
            x = torch.cat((cls_tokens, x), dim=1)
        if self.n_register_tokens > 0:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat((x, register_tokens), dim=1)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.n_register_tokens > 0:
            return x[:, :-self.n_register_tokens]
        return x

def interpolate_pos_encoding(x, pos_embed, cls_token=False, x_has_cls_token=False):
    if cls_token:
        cls_token = pos_embed[:, 0].unsqueeze(1)
        pos_embed = pos_embed[:, 1:]
    else:
        cls_token = None
    if x_has_cls_token:
        x_cls_token = x[:, 0].unsqueeze(1)
        x = x[:, 1:]

    if pos_embed.shape[1] == x.shape[1]:
        if cls_token is not None:
            return torch.cat((cls_token, pos_embed), dim=1)
        return pos_embed
    
    npatch = x.shape[1]
    N = pos_embed.shape[1]
        

    dim = x.shape[-1]
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=math.sqrt(npatch / N),
        mode='bicubic',
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    if cls_token is None:
        return pos_embed
    return torch.cat((cls_token, pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}
