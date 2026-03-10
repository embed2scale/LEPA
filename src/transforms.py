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

from logging import getLogger

from PIL import ImageFilter

import torch
import numpy as np
import torchvision.transforms as transforms
import math
import torch.nn as nn

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_resize_and_rotate(
        sample: torch.Tensor,
        output_shape: list,
        rescale_chance:float=.5,
        rotate_chance:float=.5,
        translate_chance:float=.5,
        interpolation_mode=transforms.InterpolationMode.BILINEAR,
    ) -> torch.Tensor:
    """ Randomly resize and rotate the sample and return sample and augmentation parameters. """

    sample = sample.squeeze(2)
    assert sample.shape[-2] == sample.shape[-1], \
        f"Expected square sample, got {sample.shape[-2]}x{sample.shape[-1]}." 


    rescale = True if torch.rand(1).item() < rescale_chance else False
    rotate = True if torch.rand(1).item() < rotate_chance else False
    translate = True if torch.rand(1).item() < translate_chance else False
    # translate = False
    # rescale = False
    # rotate  = True

    # Initialize parameters
    angle = 0
    scale = 1.0
    tx, ty = 0, 0
    size = (sample.shape[-2], sample.shape[-1])  # (H, W)
    fill = 0.0 # should not be needed

    if translate:
        # Randomly translate the sample
        max_translate = int((sample.shape[-1]-output_shape[-1]) / 2)
        # print(f"Max translation: {max_translate}")
        tx = torch.randint(-max_translate, max_translate+1, (1,)).item()
        ty = torch.randint(-max_translate, max_translate+1, (1,)).item()
        tmax = max(abs(tx), abs(ty))
        size = (size[0] - 2*tmax, size[1] - 2*tmax)
        

        # TODO: For continuous translation: Not tested yet!
        # sample = T.functional.affine(
        #     sample, angle=0, translate=(tx, ty), scale=1.0, shear=0,
        #     interpolation=T.InterpolationMode.BILINEAR, fill=fill)

        sample = torch.roll(sample, shifts=(tx, ty), dims=(-2, -1))
        sample = transforms.functional.center_crop(sample, size)

    assert size[0] >= output_shape[-2] and size[1] >= output_shape[-1], \
        f"Sample size {size} is smaller than target size {output_shape[-2], output_shape[-1]}."

    # Randomly resize the sample
    if rescale:
        size_before = size
        size = [torch.randint(output_shape[-2], size[0] + 1, (1,)).item()]*2
        scale = size[0] / size_before[0]
        sample = transforms.functional.resize(sample, size, interpolation=interpolation_mode)

    assert size[0] >= output_shape[-2] and size[1] >= output_shape[-1], \
        f"Sample size {size} is smaller than target size {output_shape[-2], output_shape[-1]}."

    # Randomly rotate the sample
    if rotate:
        if size[-1] == output_shape[-1] and size[-2] == output_shape[-2]:
            max_angle = 0
        else:
            max_angle = abs((np.arcsin((size[-1]-1)/output_shape[-1]/np.sqrt(2)) - np.pi / 4) * 180 / np.pi)
        # print(f"Max angle: {max_angle}")
        if np.isnan(max_angle):
            max_angle = 90
            
        if not np.isclose(max_angle, 0):
            angle = torch.empty(1).uniform_(-max_angle, max_angle).item()

            sample = transforms.functional.rotate(sample, angle, expand=False, interpolation=interpolation_mode, fill=fill)
        
    # Crop the sample to the desired size
    sample = transforms.functional.center_crop(sample, (output_shape[-2], output_shape[-1]))

    return sample, {'angle': torch.tensor([[angle]]).repeat(sample.shape[0],1), 
                    'scale': torch.tensor([[scale]]).repeat(sample.shape[0],1), 
                    'tx': torch.tensor([[tx]]).repeat(sample.shape[0],1), 
                    'ty': torch.tensor([[ty]]).repeat(sample.shape[0],1)
                    }


def get_transformed_grid(H, W, tx, ty, theta_deg, scale, patch_size=16, absolute_coords=True):
    assert tx.shape == ty.shape == theta_deg.shape == scale.shape, "Dimensions must match for tx, ty, theta_deg, and scale" 
    tx = tx.squeeze(1)
    ty = ty.squeeze(1)
    theta_deg = theta_deg.squeeze(1)
    scale = scale.squeeze(1)

    assert tx.ndim == 1, "tx must be a 1D tensor"
    assert ty.ndim == 1, "ty must be a 1D tensor"
    assert theta_deg.ndim == 1, "theta_deg must be a 1D tensor"
    assert scale.ndim == 1, "scale must be a 1D tensor"

    b = tx.shape[0]
    theta_rad = theta_deg * math.pi / 180.0

    # normalize tx,ty to [-1, 1]
    tx_norm = (tx / patch_size / W) * 2 - 1
    ty_norm = (ty / patch_size / H) * 2 - 1

    cos_t = torch.cos(theta_rad) * scale
    sin_t = torch.sin(theta_rad) * scale

    # Affine matrix for a batch
    affine_matrix = torch.zeros((b, 2, 3), dtype=cos_t.dtype, device=cos_t.device)
    affine_matrix[:, 0, 0] = cos_t
    affine_matrix[:, 0, 1] = -sin_t
    affine_matrix[:, 0, 2] = tx_norm
    affine_matrix[:, 1, 0] = sin_t
    affine_matrix[:, 1, 1] = cos_t
    affine_matrix[:, 1, 2] = ty_norm

    grid = nn.functional.affine_grid(affine_matrix, size=[b, 1, H, W], align_corners=False)  # (b, H, W, 2)
    if absolute_coords:
        grid[..., 0] = (grid[..., 0] + 1) * 0.5 * (W - 1)
        grid[..., 1] = (grid[..., 1] + 1) * 0.5 * (H - 1)
    else:
        grid[..., 0] = (grid[..., 0] + 1)
        grid[..., 1] = (grid[..., 1] + 1)
        

    return grid  # Shape: (b, H, W, 2)

def resize_and_rotate(
        sample: torch.Tensor,
        output_shape: list,
        angle:float,
        scale:float,
        tx:int,
        ty:int,
        interpolation_mode=transforms.InterpolationMode.BILINEAR,
    ) -> torch.Tensor:
    """ Resize and rotate the sample with the given parameters and return sample. """

    sample = sample.squeeze(2)
    assert sample.shape[-2] == sample.shape[-1], \
        f"Expected square sample, got {sample.shape[-2]}x{sample.shape[-1]}."
    
    import torch.nn.functional as F

    # Assume sample shape: (B, C, H, W)
    B, C, H, W = sample.shape
    patch_size = 1  # Assuming pixel-wise, adjust if needed
    target_size = output_shape[-1]
    input_size = H  # Assuming square input

    # Prepare grid
    grid = get_transformed_grid(
        H=target_size // patch_size,
        W=target_size // patch_size,
        patch_size=patch_size,
        theta_deg=angle,
        scale=scale,
        tx=tx,
        ty=ty,
        absolute_coords=True,
    )  # (B, H, W, 2) or (H, W, 2)

    # If grid is not batched, expand for batch
    if grid.dim() == 3:
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Normalize grid to [-1, 1]
    grid[..., 0] = grid[..., 0] * 2 / (input_size // patch_size - 1)
    grid[..., 1] = grid[..., 1] * 2 / (input_size // patch_size - 1)

    # grid_sample expects (B, C, H, W)
    out = F.grid_sample(
        sample,
        grid,
        mode=interpolation_mode.name.lower(),
        padding_mode='zeros',
        align_corners=True
    )

    return out
