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

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def copy_pos_embed(target_model, state_dict):
    if hasattr(target_model, 'pos_embed'):
        logger.info("Target model has pos_embed attribute")
        if 'pos_embed' in state_dict:
            state_dict['pos_embed'] = target_model.pos_embed
            logger.info("Updated pos_embed in state_dict to match target_model's pos_embed")
    if hasattr(target_model, 'predictor_pos_embed'):
        logger.info("Target model has predictor_pos_embed attribute")
        if 'predictor_pos_embed' in state_dict:
            state_dict['predictor_pos_embed'] = target_model.predictor_pos_embed
            logger.info("Updated predictor_pos_embed in state_dict to match target_model's predictor_pos_embed")
    return state_dict


def strip_module_prefix(target_model,state_dict):
    """Strips the 'module.' prefix from state_dict keys to match models not wrapped in nn.DataParallel."""
    
    logger.info(f"Stripping 'module.' prefix from state_dict keys for {target_model.__class__.__name__}")
    if 'DataParallel' not in target_model.__class__.__name__:
        # If the model is wrapped in DataParallel, we need to strip the 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    return state_dict


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        pretrained_dict = strip_module_prefix(encoder, pretrained_dict)
        pretrained_dict = copy_pos_embed(encoder, pretrained_dict)
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        pretrained_dict = strip_module_prefix(predictor, pretrained_dict)
        pretrained_dict = copy_pos_embed(predictor, pretrained_dict)
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            pretrained_dict = strip_module_prefix(target_encoder, pretrained_dict)
            pretrained_dict = copy_pos_embed(target_encoder, pretrained_dict)
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    full_img_size=256,
    crop_size=224,
    in_chans=3,
    pred_depth=6,
    pred_emb_dim=384,
    interp_pos_encoding='conditional', # conditional or interpolate
    num_conditionings=4,
    enc_has_cls_token=False,
    enc_n_register_tokens=0,
    pred_n_register_tokens=0,
    no_condition_mlp=False,
):
    encoder = vit.__dict__[model_name](
        img_size=[full_img_size],
        crop_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
        interp_pos_encoding=interp_pos_encoding,
        cls_token=enc_has_cls_token,
        n_register_tokens=enc_n_register_tokens
        )
    predictor = vit.__dict__['vit_predictor'](
        num_patches=(full_img_size // patch_size)**2,
        num_crop_patches=(crop_size // patch_size)**2,
        patch_size=patch_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        interp_pos_encoding=interp_pos_encoding,
        num_conditionings=num_conditionings,
        enc_has_cls_token=enc_has_cls_token,
        n_register_tokens=pred_n_register_tokens,
        no_condition_mlp=no_condition_mlp,)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
