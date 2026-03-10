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

import os

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel
import src.utils.metrics as metrics

from src.masks.multiblock import MaskCollator, RandomMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    Logger,
    gpu_timer,
    grad_logger,
    AverageMeter,
    get_param_norm_to_update_ratio)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.datasets import get_dataloader

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import random_resize_and_rotate
from concurrent.futures import ThreadPoolExecutor, as_completed

# --
log_timings = True
log_freq = 10
checkpoint_freq = 5
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def sync_output_dir(output_dir):
    """
    Syncs the output directory across all processes.
    This is useful when running distributed training to ensure that all processes
    have access to the same output directory.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    rank = torch.distributed.get_rank()

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory {output_dir} does not exist on rank {rank}.")
    
    logger.info(f"Output directory {output_dir} synced across all processes.")

def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']

    # -- DATA
    # use_gaussian_blur = args['data']['use_gaussian_blur']
    # use_horizontal_flip = args['data']['use_horizontal_flip']
    # use_color_distortion = args['data']['use_color_distortion']
    # color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    chunk_size = args['data']['chunk_size']
    bands = args['data']['bands']
    scaling = args['data']['scaling']
    mean = args['data']['mean']
    std = args['data']['std']
    input_size = args['data']['input_size'][-1] if isinstance(args['data']['input_size'], list) else args['data']['input_size']
    assert input_size % args['mask']['patch_size'] == 0, \
        f"Input size {input_size} must be divisible by patch size {args['mask']['patch_size']}."

    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    # image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    if isinstance(crop_size, int):
        assert crop_size % args['mask']['patch_size'] == 0, \
            f"Crop size {crop_size} must be divisible by patch size {args['mask']['patch_size']}."
    else:
        assert crop_size[-1] % args['mask']['patch_size'] == 0, \
            f"Crop size {crop_size[-1]} must be divisible by patch size {args['mask']['patch_size']}."

    max_images = args['data'].get('max_images', None)
    max_eval_images = args['data'].get('max_eval_images', None)
    # --

    # -- MODEL
    model_name = args['model']['model_name_enc']
    interp_pos_encoding = args['model']['interp_pos_encoding']  # 'conditional' or 'interpolate'
    pred_depth = args['model']['pred_depth']
    pred_emb_dim = args['model']['pred_emb_dim']
    condition_on: list[str] = args['model']['condition_on']
    enc_has_cls_token = args['model']['enc_has_cls_token']
    enc_n_register_tokens = args['model']['enc_n_register_tokens']
    pred_n_register_tokens = args['model']['pred_n_register_tokens']
    finetune_only_predictor = args['model'].get('finetune_only_predictor', False)

    # -- MASK
    collator_type = args['mask']['collator_type']  # 'multiblock' or 'random'
    mask_predictions = args['mask']['mask_predictions']  # whether to predict masks or not
    mask_context = args['mask']['mask_context']  # whether to mask context blocks or not
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    jepa_target = args['mask']['jepa_target']  # whether to use JEPA target (center crop of input image) or not
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    loss_function = args['optimization']['loss']  # loss function to use
    loss_function = getattr(metrics, loss_function) if isinstance(loss_function, str) else loss_function

    # -- LOGGING
    folder = args['logging']['folder'].rstrip(os.sep)
    i=1
    while not resume_preempt and os.path.exists(folder):
        folder = f"{args['logging']['folder'].rstrip(os.sep)}_{i}"
        i += 1
    args['logging']['folder'] = folder

    tag = args['logging']['write_tag']
    use_wandb = args['logging']['use_wandb']
    wandb_entity = args['logging']['wandb_entity'] if use_wandb else None
    wandb_project = args['logging']['wandb_project'] if use_wandb else None
    wandb_name = f'{args["logging"]["wandb_name"]}_{tag}_{args["jobid"]}' if use_wandb else None

    # -- EVAL
    eval_root_path = args['evaluation']['eval_root_path']
    eval_metrics = args['evaluation']['eval_metrics']
    eval_metrics = [(name, getattr(metrics, name)) for name in eval_metrics]

    
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank, device = init_distributed()
    # print(f'Using device: {device}, rank: {rank}, world_size: {world_size}, cudavisible_devices: {os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")}')
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        # logger.setLevel(logging.ERROR)
        pass

    # -- sync output directory and dump config
    sync_output_dir(folder)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    os.makedirs(folder, exist_ok=True)
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        if resume_preempt:
            # -- resume preempted training
            if r_file is not None and os.path.sep in r_file:
                load_path = r_file
            else:
                r_file = f'{tag}-latest.pth.tar' if r_file is None else r_file
                load_path = os.path.join(folder, r_file)
            if not os.path.exists(load_path):
                raise FileNotFoundError(f'Checkpoint {load_path} does not exist.')
        else:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        full_img_size=input_size,
        crop_size=crop_size[-1],
        in_chans=len(bands),
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        interp_pos_encoding=interp_pos_encoding,
        num_conditionings=len(condition_on),
        enc_has_cls_token=enc_has_cls_token,
        enc_n_register_tokens=enc_n_register_tokens,
        pred_n_register_tokens=pred_n_register_tokens
        )
    target_encoder = copy.deepcopy(encoder)

    img_key = 'sample' if 'terramesh' not in root_path.lower() else 'image'
    target_key = 'target' if 'terramesh' not in root_path.lower() else 'image'

    # -- make data transforms and dataloaders
    if collator_type == 'multiblock':
        mask_collator = MaskCollator(
            input_size=input_size, # have to give the full size, downscale indices later when loading
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            allow_overlap=allow_overlap,
            image_key=img_key,
            target_key=target_key,
            min_keep=min_keep)
    elif collator_type == 'random':
        mask_collator = RandomMaskCollator(
            input_size=input_size,  # have to give the full size, downscale indices later when loading
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            allow_overlap=allow_overlap,
            image_key=img_key,
            target_key=target_key,
            min_context_tokens=min_keep)

    unsupervised_loader, unsupervised_sampler, eval_loader, eval_sampler, ipe, ipve = get_dataloader(
        root_path=root_path,
        eval_root_path=eval_root_path,
        mask_collator=mask_collator,
        batch_size=batch_size,
        chunk_size=chunk_size,
        input_size=input_size,
        crop_size=crop_size,
        bands=bands,
        scaling=scaling,
        mean=mean,
        std=std,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )
    ipe = min(ipe, max_images// (world_size * batch_size)) if max_images is not None else ipe
    logger.info(f'Using {ipe} iterations per epoch (max_images={max_images})')

    ipve = min(ipve, max_eval_images//(world_size*batch_size)) if max_eval_images is not None else ipve
    logger.info(f'Using {ipve} iterations for evaluation (max_imgages={max_eval_images})')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    # -- make logger
    train_logger = Logger(
        train_metrics=['loss', 'mask-A', 'mask-B', 'time (ms)'],
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        val_metrics=[name for name, _ in eval_metrics],
        entity=wandb_entity if use_wandb else None,
        project=wandb_project if use_wandb else None,
        directory=folder,
        imgs_per_epoch=ipe)
    train_logger.log_config(args)

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()
        train_logger.global_step = start_epoch * ipe
    
    if finetune_only_predictor:
        logger.info('Freezing encoder parameters, copying encoder to target encoder')
        for param in encoder.parameters():
            param.requires_grad = False
        target_encoder.load_state_dict(encoder.state_dict())

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0 or epoch < 10:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
                logger.info(f'Saved checkpoint at epoch {epoch + 1} to {save_path.format(epoch=f"{epoch + 1}")}')

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        def load_imgs(udata,masks_enc,masks_pred,mask_context=mask_context,mask_predictions=mask_predictions):
            # -- unsupervised imgs
            imgs = udata[img_key].to(device, non_blocking=True)
            augmentation_params = {k:v.to(device, non_blocking=True)
                                    for k,v in udata['augmentation_params'].items() if k in condition_on} if 'augmentation_params' in udata else {}
            # temporal_coords, location_coords, target = udata['temporal_coords'], udata['location_coords'], 
            if jepa_target:
                targets = T.functional.center_crop(imgs, crop_size[-1])
            else:
                if 'terramesh' in root_path.lower():
                    # Apply random_resize_and_rotate per image and merge augmentation params
                    targets_list = []
                    aug_list = []
                    for i_img in range(imgs.size(0)):
                        img_i = imgs[i_img:i_img+1]
                        tgt_i, aug_i = random_resize_and_rotate(img_i, crop_size)
                        targets_list.append(tgt_i)
                        # keep only requested condition keys and move to device
                        aug_list.append({k: v.to(device, non_blocking=True)
                                         for k, v in aug_i.items() if k in condition_on})

                    targets = torch.cat(targets_list, dim=0)

                    # merge augmentation params: stack tensors for each key across batch
                    augmentation_params = {
                        k: torch.cat([aug_list[i][k] for i in range(len(aug_list))], dim=0).to(device, non_blocking=True)
                        for k in aug_list[0].keys()
                    }
                else:
                    targets = udata[target_key]
            targets = targets.to(device, non_blocking=True)
            if mask_context:
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            else:
                masks_1 = [torch.arange(0, imgs.size(2)//patch_size * imgs.size(3) // patch_size, device=device).unsqueeze(0).repeat(imgs.size(0), 1)] * len(masks_enc)
            if mask_predictions:
                masks_2 = mask_collator.convert_large_mask_to_small(masks_pred, input_size//patch_size, crop_size[-1]//patch_size)
            else:
                masks_2 = [torch.arange(0, targets.size(2)//patch_size * targets.size(3) // patch_size, device=device).unsqueeze(0).repeat(imgs.size(0), 1)]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_2]
            return (imgs, targets, augmentation_params, masks_1, masks_2)

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            if itr >= ipe:
                logger.info(f'Finished epoch {epoch + 1} after {itr} iterations.')
                break
            # empty cache
            # if torch.cuda.is_available(): 
            #     torch.cuda.empty_cache() -> leads to performance increase 620ms -> 800ms per step

            imgs, targets, augmentation_params, masks_enc, masks_pred = load_imgs(udata, masks_enc, masks_pred)
            assert imgs.size(2) == input_size and imgs.size(3) == input_size, \
                f'Input size {imgs.size()} does not match expected size {(imgs.size(0), len(bands), input_size, input_size)}'
            assert targets.size(2) == crop_size[-1] and targets.size(3) == crop_size[-1], \
                f'Target size {targets.size()} does not match expected size {(targets.size(0), len(bands), crop_size[-1], crop_size[-1])}'

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():# -> tuple[float, Any | float, float, AverageMeter, dict[int, ...:
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(targets)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        if enc_has_cls_token:
                            cls_token = h[:,0:1,:]
                            h = h[:,1:,:]
                        h = apply_masks(h, masks_pred)
                        if enc_has_cls_token:
                            h = torch.cat([cls_token.repeat(len(masks_pred),1,1), h], dim=1)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    if finetune_only_predictor:
                        with torch.no_grad():
                            context = encoder(imgs, masks_enc)
                    else:
                        context = encoder(imgs, masks_enc)
                    z = predictor(context, masks_enc, masks_pred, conditions = augmentation_params)
                    return z, context

                def loss_fn(z, h, context):
                    if enc_has_cls_token:
                        cls_token = h[:,0:1,:]
                        h = h[:,1:,:]
                    if enc_has_cls_token:
                        context_cls_token = context[:,0:1,:]
                        context_cls_token = context_cls_token.repeat(len(masks_pred),1,1)
                    loss = loss_function(z, h)
                    # if enc_has_cls_token: # eg DINO has additional cls head, but we dont have the same different views
                    #     cls_loss = loss_function(context_cls_token, cls_token)
                    #     loss = loss + cls_loss
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.amp.autocast('cuda',dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z,context = forward_context()
                    loss = loss_fn(z, h, context)

                #  Step 2. Backward & step
                # old_named_params = [(name, p.clone().detach()) for name, p in encoder.named_parameters()]
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # norm_to_update_ratio = get_param_norm_to_update_ratio(
                #     encoder.named_parameters(), old_named_params)
                norm_to_update_ratio = {0: 0.0}  # dummy value, not used in this case
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss.detach()), _new_lr, _new_wd, grad_stats, norm_to_update_ratio)
            (loss, _new_lr, _new_wd, grad_stats, norm_to_update_ratio), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                # norm_to_update_ratio = {k: v for k, v in norm_to_update_ratio.items() if v < 1000.0}  # filter out large ratios
                norm_to_update_ratio_max = max(norm_to_update_ratio.values()) if norm_to_update_ratio else 0.0
                norm_to_update_ratio_min = min(norm_to_update_ratio.values()) if norm_to_update_ratio else 0.0
                norm_to_update_ratio_mean = np.mean(list(norm_to_update_ratio.values())) if norm_to_update_ratio else 0.0
                norm_to_update_ratio_median = np.median(list(norm_to_update_ratio.values())) if norm_to_update_ratio else 0.0
                norm_to_update_ratio_std = np.std(list(norm_to_update_ratio.values())) if norm_to_update_ratio else 0.0
                
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(f'[{epoch + 1}, {itr:5d}] loss: {loss_meter.avg:.3f} '
                                f'masks: {maskA_meter.avg:.1f} {maskB_meter.avg:.1f} '
                                f'[wd: {_new_wd:.2e}] [lr: {_new_lr:.2e}] '
                                f'[mem: {torch.cuda.max_memory_allocated() / 1024.**2:.2f} MB] [reserved: {torch.cuda.max_memory_reserved() / 1024.**2:.2f} MB] '
                                f'({time_meter.avg:.1f} ms)'
                                f' [eta: {time_meter.avg * (ipe - itr) / 1000:.1f} s]')

                    if grad_stats is not None:
                        logger.info(f'[{epoch + 1}, {itr:5d}] grad_stats: '
                                    f'[{grad_stats.first_layer:.2e} {grad_stats.last_layer:.2e}] '
                                    f'({grad_stats.min:.2e}, {grad_stats.max:.2e})')

                train_logger.log_train(epoch + 1, itr, **{
                    "loss": loss,
                    "mask-A": len(masks_enc[0][0]),
                    "mask-B": len(masks_pred[0][0]),
                    "time (ms)": etime,
                    "lr": _new_lr,
                    "wd": _new_wd,
                    "norm2update_ratio_mean": norm_to_update_ratio_mean,
                    "norm2update_ratio_median": norm_to_update_ratio_median,
                    "norm2update_ratio_std": norm_to_update_ratio_std,
                    "norm2update_ratio_max": norm_to_update_ratio_max,
                    "norm2update_ratio_min": norm_to_update_ratio_min,
                    })
                if grad_stats is not None:
                    train_logger.log_train(
                        epoch + 1, itr,
                        **{"grad_first_layer": grad_stats.first_layer,
                            "grad_last_layer": grad_stats.last_layer,
                            "grad_min": grad_stats.min,
                            "grad_max": grad_stats.max
                        })

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

        # -- EVALUATION
        logger.info('Evaluation after epoch %d' % (epoch + 1))
        eval_losses = {name: AverageMeter() for name, _ in eval_metrics}
        eval_losses = {**eval_losses, **{name+'_enc_target': AverageMeter() for name, _ in eval_metrics}}
        eval_losses = {**eval_losses, **{name+'_enc_target_cls': AverageMeter() for name, _ in eval_metrics}}
        eval_losses = {**eval_losses, **{"MRR": AverageMeter(), "MRR_var": AverageMeter()}}
        
        def evaluate_embeddings(z, h, encoded=None):
            if enc_has_cls_token:
                cls_token_target = h[:,0:1,:]
                h_patches = h[:,1:,:]
            else:
                h_patches = h
            for name,metric in eval_metrics:
                logger.info(f'Computing eval metric {name}')
                value = metric(z, h_patches)
                logger.info(f'  {name}: {value:.3f}')
                eval_losses[name].update(AllReduce.apply(value).item())
                logger.info(f'  {name} (avg): {eval_losses[name].avg:.3f}')
            if encoded is not None:
                if enc_has_cls_token:
                    cls_token_enc = encoded[:,0:1,:]
                    encoded = encoded[:,1:,:]
                for name,metric in eval_metrics:
                    value = metric(h_patches, encoded)
                    eval_losses[name+'_enc_target'].update(AllReduce.apply(value).item())
                    if enc_has_cls_token:
                        cls_diff = metric(cls_token_target, cls_token_enc)
                        eval_losses[name+'_enc_target_cls'].update(AllReduce.apply(cls_diff).item())

        @torch.no_grad()
        def evaluate_dataset():
            # enable debug logging on all ranks
            logger.info('Starting evaluation loop')
            # logger.setLevel(logging.DEBUG)
            for itr, (udata, masks_enc, masks_pred) in enumerate(eval_loader):
                
                logger.info(msg=f'Eval iteration {itr}')
                if itr > 3:
                    logger.info(f'Finished evaluation after {itr} iterations.')
                    break
                imgs, targets, augmentation_params, masks_enc, masks_pred = load_imgs(udata,masks_enc,masks_pred,mask_predictions=False,mask_context=False)
                h = target_encoder(targets)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h)
                # -- create targets (masked regions of h)
                if enc_has_cls_token:
                    cls_token = h[:,0:1,:]
                    h = h[:,1:,:]
                h = apply_masks(h, masks_pred)
                if enc_has_cls_token:
                    h = torch.cat([cls_token, h], dim=1)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred, conditions=augmentation_params)
                logger.info(f'Computing eval metrics on {len(z)} vectors of dimension {z.size(-1)}')
                encoded_targets = encoder(targets, masks_pred)
                evaluate_embeddings(z, h, encoded_targets)
                logger.info(f'Finished eval iteration {itr}')

                if itr<2: # time consuming, so only do for first few batches
                    logger.info('Computing MRR metric')
                    mrr, mrr_var = metrics.mean_reciprocal_rank(
                        encoder,
                        predictor,
                        imgs[:64],
                        patch_size,
                        crop_size[-1],
                        condition_on,
                        device,
                        interpolate_not_predict=jepa_target
                    )
                    eval_losses["MRR"].update(AllReduce.apply(mrr).item())
                    logger.info(f'  MRR: {mrr:.3f} (avg: {eval_losses["MRR"].avg:.3f})')
                    eval_losses["MRR_var"].update(AllReduce.apply(mrr_var).item())
                    logger.info(f'  MRR_var: {mrr_var:.3f} (avg: {eval_losses["MRR_var"].avg:.3f})')


            # if rank>0:
            #     logger.setLevel(logging.ERROR)
            # else:
            #     logger.setLevel(logging.INFO)

        # logger.info('Running evaluation dataset')
        # if rank == 0:
        #     _, etime = gpu_timer(evaluate_dataset)
        # torch.distributed.barrier()
        # logger.info('Finished evaluation dataset')


        # -- log eval metrics
        # for name, _ in eval_losses.items():
        #     logger.info(f'Eval {name}: {eval_losses[name].avg:.3f} ({etime/1000:.1f} s)')
        # train_logger.log_val(epoch + 1, **{name: eval_losses[name].avg for name in eval_losses}, **{"time (ms)": etime})

if __name__ == "__main__":
    # args are in --fname argument
    import argparse
    import pprint
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='configs.yaml')
    parser.add_argument(
        '--resume_preempt', action='store_true',
        help='whether to resume preempted training')
    parser.add_argument(
        '--jobid', type=str, default=str(random.randint(100000, 999999)),
        help='job id for logging purposes. Defauls to random number if not set')
    parser.add_argument(
        '--data__root_path', type=str, default=None,
        help='root path for dataset.')
    parser.add_argument(
        '--evaluation__eval_root_path', type=str, default=None,
        help='root path for evaluation dataset.')
    parser.add_argument(
        '--data__bands', type=str, nargs='+', default=None,
        help='list of bands to use for the model. e.g. ["B02", "B03", "B04", "B05", "B06", "B07"].')
    parser.add_argument(
        '--data__num_workers', type=int, default=None,
        help='number of workers to use for data loading.')
    parser.add_argument(
        '--data__batch_size', type=int, default=None,
        help='batch size to use for training.')
    parser.add_argument(
        '--data__input_size', type=int, nargs='+', default=None,
        help='input size for the model. Should be an integer or list of integers, e.g. 1 224 224 for single time step.')
    parser.add_argument(
        '--data__crop_size', type=int, nargs='+', default=None,
        help='crop size for the model. Should be an integer or list of integers, e.g. 1 224 224 for single time step.')
    parser.add_argument(
        '--data__max_images', type=int, default=None,
        help='maximum number of images to use for training. If None, use all images.') 
    parser.add_argument(
        '--data__max_eval_images', type=int, default=None,
        help='maximum number of images to use for evaluation. If None, use all images.') 
    parser.add_argument(
        '--logging__folder', type=str, default=None,
        help='output folder for logging and checkpoints.')
    parser.add_argument(
        '--logging__wandb_name', type=str, default=None,
        help='wandb name for logging.')
    parser.add_argument(
        '--logging__write_tag', type=str, default=None,
        help='tag to use for logging and checkpoints.')
    parser.add_argument(
        '--mask__collator_type', type=str, default=None,
        help='type of mask collator to use. Options: multiblock, random.')
    parser.add_argument(
        '--mask__patch_size', type=int, default=None,
        help='patch size for the model. Should be an integer, e.g. 14 for 14x14 patches.')
    parser.add_argument(
        '--mask__jepa_target', type=lambda x: x.lower() == 'true', default=None,
        help='whether to use JEPA target (center crop of input image) or not.')
    parser.add_argument(
        '--mask__allow_overlap', type=lambda x: x.lower() == 'true', default=None,
        help='whether to allow overlap between context and target blocks.')
    parser.add_argument(
        '--mask__enc_mask_scale', type=float, nargs='+', default=None,
        help='Range of scales for context blocks. e.g. 0.1 0.2 for 10% to 20% of the image size.')
    parser.add_argument(
        '--mask__pred_mask_scale', type=float, nargs='+', default=None,
        help='Range of scales for target blocks. e.g. 0.1 0.2 for 10% to 20% of the image size.')
    parser.add_argument(
        '--mask__mask_predictions', type=lambda x: x.lower() == 'true', default=None,
        help='whether to mask target blocks or not.')
    parser.add_argument(
        '--mask__mask_context', type=lambda x: x.lower() == 'true', default=None,
        help='whether to mask context blocks or not.')
    parser.add_argument(
        '--model__model_name_enc', type=str, default=None,
        help='name of the model to use for the encoder. e.g. vit_huge, vit_base, etc.')
    parser.add_argument(
        '--model__condition_on', type=str, nargs='+', default=None,
        help='list of conditionings to use for the model. e.g. temporal_coords, location_coords, etc.')
    parser.add_argument(
        '--model__interp_pos_encoding', type=str, default=None,
        help='interpolation method for positional encoding. Options: conditional, interpolate.')
    parser.add_argument(
        '--model__pred_depth', type=int, default=None,
        help='number of transformer blocks in the predictor.')
    parser.add_argument(
        '--model__pred_emb_dim', type=int, default=None,
        help='embedding dimension for the predictor. e.g. 384 if you want to introduce bottleneck.')
    parser.add_argument(
        '--model__enc_has_cls_token', type=lambda x: x.lower() == 'true', default=None,
        help='whether the encoder has a class token or not. If True, the encoder will output a class token.')
    parser.add_argument(
        '--model__enc_n_register_tokens', type=int, default=None,
        help='number of register tokens for the encoder. e.g. 1 for a single register token.')
    parser.add_argument(
        '--model__pred_n_register_tokens', type=int, default=None,
        help='number of register tokens for the predictor. e.g. 1 for a single register token.')
    parser.add_argument(
        '--model__finetune_only_predictor', type=lambda x: x.lower() == 'true', default=None,
        help='whether to train only the predictor and keep the encoder frozen.')
    parser.add_argument(
        '--optimization__epochs', type=int, default=None,
        help='number of epochs to train for.')
    parser.add_argument(
        '--optimization__lr', type=float, default=None,
        help='learning rate to use for training.')
    parser.add_argument(
        '--optimization__start_lr', type=float, default=None,
        help='starting learning rate to use for training.')
    parser.add_argument(
        '--optimization__final_lr', type=float, default=None,
        help='final learning rate to use for training.')
    parser.add_argument(
        '--meta__read_checkpoint', type=str, default=None,
        help='checkpoint file to read. If None, use latest checkpoint.')
    args = parser.parse_args()
    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        # merge args into params
        for k, v in vars(args).items():
            if v is None:
                continue
            if '__' in k:
                first, second = k.split('__', 1)
                if first in params and isinstance(params[first], dict):
                    params[first][second] = v
                else:
                    if first not in params:
                        params[first] = {}
                        params[first][second] = v
            else:
                params[k] = v
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    main(args=params,
         resume_preempt=args.resume_preempt)
