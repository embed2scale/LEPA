# Compute the Mean Reciprocal Rank (MRR) on the entire eval dataset using distributed data parallel (torchrun).
# This cell assumes the notebook is run with torchrun and the config path is provided as a command-line argument.


# run with torchrun --nnodes=1 --nproc_per_node=4 ./eval_MRR_fulldataset.py --config logs/iwm_vitb16_hls_256192_cls_corrposcond_correctedcls_nolosschange/params-ijepa.yaml
# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 50 --config logs/jepa_vitb_imagenet_224_register/params-ijepa.yaml > logs/jepa_vitb_imagenet_224_register/eval_MRR.log


# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 55  --sample_mode nearest --config logs/finetune_iwm_vitb16_hls_256192_predEmb768_corr/params-ijepa.yaml > logs/finetune_iwm_vitb16_hls_256192_predEmb768_corr/eval_MRR_nearest.log
# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 55  --sample_mode nearest --config logs/finetune_iwm_vitb16_hls_256192_predEmb192_corr/params-ijepa.yaml > logs/finetune_iwm_vitb16_hls_256192_predEmb192_corr/eval_MRR_nearest.log


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import re

from src.masks.multiblock import MaskCollator, RandomMaskCollator
from src.datasets.datasets import get_dataloader
from src.helper import load_checkpoint, init_model
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
import argparse
from src.utils.logging import AverageMeter
import time
import src.utils.metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='Path to config file')
    parser.add_argument('--epoch', type=str, default='50', help='Checkpoint epoch')
    parser.add_argument('--n_aug', type=int, default=256, help='Number of augmentations per sample')
    parser.add_argument('--pangaea_model', type=str, default='')
    parser.add_argument('--sample_mode', type=str, default='bilinear', help='Sampling mode for interpolation: bilinear or nearest')
    parser.add_argument('--interpolate', action='store_true', help='Whether to interpolate predictions instead of predicting')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output')
    return parser.parse_args()

args = parse_args()

if args.pangaea_model: 
    from hydra.utils import instantiate
    from pangaea.encoders.base import Encoder
    with open(f'../pangaea-bench/configs/encoder/{args.pangaea_model}.yaml') as f:
        model_config = yaml.safe_load(f)

    if 'num_frames' in model_config:
        model_config['num_frames']=1
    model_bands = model_config.get('input_bands', None).get('optical', None)
    # Override input_size for pangaea models to match data
    model_config['input_size']=256
    # img_size is not supported by all models (e.g., Prithvi), so only set if it exists in config
    if 'img_size' in model_config:
        model_config['img_size']=256
    # For models that use input_size instead of img_size, also set img_size to match
    # But only if the encoder supports it (check after instantiation)

    encoder: Encoder = instantiate(model_config)
    encoder.eval()
    
    # Check if model has encoder_embeddings (Terramind) or single embedding (Prithvi)
    has_encoder_embeddings = hasattr(encoder, 'encoder_embeddings')
    
    # Interpolate positional embeddings if img_size differs from checkpoint
    # The checkpoint has positional embeddings for the pretrained img_size (224)
    # We need to interpolate them to match the current model's img_size (256)
    import torch.nn.functional as F
    import math
    
    # Get the checkpoint path from model_config
    checkpoint_path = model_config.get('encoder_weights', None)
    checkpoint = None  # Initialize checkpoint to None
    if checkpoint_path is not None:
        # Load the checkpoint to get the original positional embeddings
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Interpolate positional embeddings based on model type
        if has_encoder_embeddings:
            # Terramind: has encoder_embeddings with multiple modalities
            for mod_name, emb_module in encoder.encoder_embeddings.items():
                if hasattr(emb_module, 'pos_emb'):
                    checkpoint_pos_emb_key = f"encoder_embeddings.{mod_name}.pos_emb"
                    if checkpoint_pos_emb_key in checkpoint:
                        checkpoint_pos_emb = checkpoint[checkpoint_pos_emb_key]
                        current_pos_emb = emb_module.pos_emb
                        
                        # If shapes differ, interpolate
                        if checkpoint_pos_emb.shape[1] != current_pos_emb.shape[1]:
                            print(f"Interpolating positional embeddings for {mod_name}: "
                                  f"{checkpoint_pos_emb.shape[1]} -> {current_pos_emb.shape[1]}")
                            
                            # Get original h, w and new h, w
                            num_pos = checkpoint_pos_emb.shape[1]
                            sqrt_num_pos = int(math.sqrt(num_pos))
                            checkpoint_h = sqrt_num_pos
                            checkpoint_w = sqrt_num_pos
                            
                            current_num_pos = current_pos_emb.shape[1]
                            sqrt_current_num_pos = int(math.sqrt(current_num_pos))
                            current_h = sqrt_current_num_pos
                            current_w = sqrt_current_num_pos
                            
                            # Reshape and interpolate
                            checkpoint_pos_emb_reshaped = checkpoint_pos_emb.reshape(1, checkpoint_h, checkpoint_w, -1)
                            checkpoint_pos_emb_reshaped = checkpoint_pos_emb_reshaped.permute(0, 3, 1, 2)
                            
                            interpolated_pos_emb = F.interpolate(
                                checkpoint_pos_emb_reshaped,
                                size=(current_h, current_w),
                                mode="bicubic",
                                align_corners=False,
                            )
                            
                            interpolated_pos_emb = interpolated_pos_emb.permute(0, 2, 3, 1).reshape(1, -1, current_pos_emb.shape[2])
                            
                            # Update the model's positional embeddings
                            # Since pos_emb is already registered as a buffer, we need to update it directly
                            emb_module.pos_emb = interpolated_pos_emb
                        else:
                            print(f"Positional embeddings already match for {mod_name}, skipping interpolation")
        else:
            # Prithvi: has single pos_embed attribute
            if hasattr(encoder, 'pos_embed'):
                checkpoint_pos_emb_key = "pos_embed"
                if checkpoint_pos_emb_key in checkpoint:
                    checkpoint_pos_emb = checkpoint[checkpoint_pos_emb_key]
                    current_pos_emb = encoder.pos_embed
                    
                    # If shapes differ, interpolate
                    if checkpoint_pos_emb.shape[1] != current_pos_emb.shape[1]:
                        print(f"Interpolating positional embeddings for Prithvi: "
                              f"{checkpoint_pos_emb.shape[1]} -> {current_pos_emb.shape[1]}")
                        
                        # Get original h, w and new h, w
                        num_pos = checkpoint_pos_emb.shape[1]
                        sqrt_num_pos = int(math.sqrt(num_pos))
                        checkpoint_h = sqrt_num_pos
                        checkpoint_w = sqrt_num_pos
                        
                        current_num_pos = current_pos_emb.shape[1]
                        sqrt_current_num_pos = int(math.sqrt(current_num_pos))
                        current_h = sqrt_current_num_pos
                        current_w = sqrt_current_num_pos
                        
                        # Reshape and interpolate
                        checkpoint_pos_emb_reshaped = checkpoint_pos_emb.reshape(1, checkpoint_h, checkpoint_w, -1)
                        checkpoint_pos_emb_reshaped = checkpoint_pos_emb_reshaped.permute(0, 3, 1, 2)
                        
                        interpolated_pos_emb = F.interpolate(
                            checkpoint_pos_emb_reshaped,
                            size=(current_h, current_w),
                            mode="bicubic",
                            align_corners=False,
                        )
                        
                        interpolated_pos_emb = interpolated_pos_emb.permute(0, 2, 3, 1).reshape(1, -1, current_pos_emb.shape[2])
                        
                        # Update the model's positional embeddings
                        encoder.pos_embed = interpolated_pos_emb
                    else:
                        print(f"Positional embeddings already match for Prithvi, skipping interpolation")

    
    config = {'mask':{},'data':{},'evaluation':{},'model':{}}
    config['data']['input_size'] = 256
    config['data']['crop_size'] = 192
    config['mask']['patch_size'] = model_config["patch_size"]
    config['data']['batch_size'] = 32
    config['data']['root_path'] = "/p/scratch/geofm4eo/HLSv9/train"
    config['mask']['collator_type'] = "multiblock"
    config['mask']['pred_mask_scale'] = [.15,.2]
    config['mask']['enc_mask_scale'] = [.85,1.]
    config['mask']['aspect_ratio'] = [.75,1.5]
    config['mask']['num_enc_masks'] = 1
    config['mask']['num_pred_masks'] = 1
    config['mask']['allow_overlap'] = False
    config['mask']['min_keep'] = 10
    config['data']['bands'] = ["B02", "B03", "B04", "B05", "B06", "B07"]
    config['data']['mean'] = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    config['data']['std'] = [2248.0,2179.0,2178.0,1850.0,1242.0,1049.0]
    config['evaluation']['eval_root_path'] = "/p/scratch/geofm4eo/HLSv9/val"
    config['data']['chunk_size'] = 16
    config['data']['scaling'] = "standard"
    config['data']['num_workers'] = 8
    config['data']['pin_mem'] = True
    config['mask']['jepa_target'] = True
    config['model']['condition_on'] = ["angle","scale","tx","ty"]
else:
    config_path = args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(stream=f)

    model_bands = config['data']['bands'] # pangaea compatibility

# Set input_size based on whether we're using a pangaea model
if args.pangaea_model:
    input_size = 256
else:
    input_size = config['data']['input_size'][-1] if isinstance(config['data']['input_size'], list) else config['data']['input_size']

epoch = args.epoch
n_aug = args.n_aug
# Distributed setup
world_size,local_rank,device=init_distributed()
# torch.cuda.set_device(local_rank)


crop_size = config['data']['crop_size'][-1] if isinstance(config['data']['crop_size'], list) else config['data']['crop_size']
patch_size = config['mask']['patch_size']
batch_size = config['data']['batch_size']
image_key = 'sample' if 'hls' in config['data']['root_path'].lower() or 'net' in config['data']['root_path'] else 'image'

if not 'collator_type' in config['mask']:
    config['mask']['collator_type'] = 'multiblock'
if config['mask']['collator_type'] == 'multiblock':
    mask_collator = MaskCollator(
        input_size=input_size,
        patch_size=patch_size,
        pred_mask_scale=config['mask']['pred_mask_scale'],
        enc_mask_scale=config['mask']['enc_mask_scale'],
        aspect_ratio=config['mask']['aspect_ratio'],
        nenc=config['mask']['num_enc_masks'],
        npred=config['mask']['num_pred_masks'],
        allow_overlap=config['mask']['allow_overlap'],
        min_keep=config['mask']['min_keep'],
        image_key=image_key
    )
elif config['mask']['collator_type'] == 'random':
    mask_collator = RandomMaskCollator(
        input_size=input_size,
        patch_size=patch_size,
        pred_mask_scale=config['mask']['pred_mask_scale'],
        enc_mask_scale=config['mask']['enc_mask_scale'],
        aspect_ratio=config['mask']['aspect_ratio'],
        nenc=config['mask']['num_enc_masks'],
        npred=config['mask']['num_pred_masks'],
        allow_overlap=config['mask']['allow_overlap'],
        min_context_tokens=config['mask']['min_keep'],
    )

if 'hls' in config['data']['root_path'].lower():
    bands = config['data']['bands']  
    mean = config['data']['mean']
    std = config['data']['std']
elif 'net' in config['data']['root_path'].lower():
    bands = ['B04','B03','B02']
    mean = config['data']['mean'][0:3][::-1]
    std = config['data']['std'][0:3][::-1]
elif 'terra' in config['data']['root_path'].lower():
    # raise NotImplementedError("TerraMesh not implemented yet (more than the HLS bands)")
    bands = ['1','2','3','4','5','6','7','8','9','0','1','2']
    mean = 0
    std = 0
else:
    bands = config['data']['bands']
    mean = config['data']['mean']
    std = config['data']['std']

print(f"Using bands: {bands}")
print(f"Using mean: {mean}")
print(f"Using std: {std}")

train_loader, _, eval_loader, _, ipe, ipve = get_dataloader(
    root_path='/p/scratch/geofm4eo/HLSv9/train',#config['data']['root_path'],
    eval_root_path='/p/scratch/geofm4eo/HLSv9/val',#config['evaluation']['eval_root_path'],
    mask_collator=mask_collator,
    batch_size=batch_size,
    chunk_size=config['data']['chunk_size'],
    input_size=input_size,
    crop_size=crop_size,
    bands=bands,
    scaling=config['data']['scaling'],
    mean=mean,
    std=std,
    world_size=world_size,
    rank=local_rank,
    num_workers=config['data']['num_workers'],
    pin_mem=config['data']['pin_mem'],
    shuffle=False,
    crop_scale=(1.0, 1.0)
)


if args.pangaea_model:
    predictor=encoder
else:
    if epoch != 'latest':
        checkpoint_path = os.path.join(os.path.dirname(config_path), f'{config["logging"]["write_tag"]}-ep{epoch}.pth.tar')
    else:
        checkpoint_path = os.path.join(os.path.dirname(config_path), f'{config["logging"]["write_tag"]}-latest.pth.tar')
    assert os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        full_img_size=input_size,
        crop_size=crop_size,
        in_chans=len(bands),
        pred_depth=config['model']['pred_depth'],
        pred_emb_dim=config['model']['pred_emb_dim'],
        model_name=config['model']['model_name_enc'],
        interp_pos_encoding=config['model']['interp_pos_encoding'],
        num_conditionings=len(config['model']['condition_on']),
        enc_has_cls_token=config['model'].get('enc_has_cls_token', False),
        enc_n_register_tokens=config['model'].get('enc_n_register_tokens', 0),
        pred_n_register_tokens=config['model'].get('pred_n_register_tokens', 0),
        no_condition_mlp=config['model'].get('no_condition_mlp', False),
    )
    # Load checkpoint and save the dictionary for selective band loading
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder, predictor, _, _, _, _ = load_checkpoint(
        device=device,
        r_path=checkpoint_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=None,
        opt=None,
        scaler=None,
    )
encoder.to(device)
predictor.to(device)

# Selectively load band weights based on available data bands
# This ensures only weights for bands that exist in both data and model get loaded
# Missing bands get random initialization instead of zero padding
data_bands_raw = config['data']['bands']  # e.g., ["B02", "B03", "B04", "B05", "B06", "B07"]

# Get model optical bands - different for pangaea vs non-pangaea models
if args.pangaea_model:
    model_bands_optical = model_config.get('input_bands', None).get('optical', None)
else:
    # For non-pangaea models, assume all bands are present (use data bands directly)
    model_bands_optical = data_bands_raw

# Special handling for TerraMind: map data bands to correct model bands based on spectral content
# The data contains B02, B03, B04, B05, B06, B07 but these correspond to different spectral bands
# B05, B06, B07 in the data actually contain NIR/SWIR information (B08A, B11, B12)
# This mapping ensures correct weight transfer for TerraMind models
terramind_band_mapping = None
if args.pangaea_model == 'terramind_base' or args.pangaea_model == 'terramind_large':
    # Data bands: B02, B03, B04, B05, B06, B07
    # These correspond spectrally to: Blue, Green, Red, B08A, B11, B12
    # Model expects: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B10, B11, B12
    # Create mapping from data band index to model band index
    # B02(0) -> B2(1), B03(1) -> B3(2), B04(2) -> B4(3), B05(3) -> B8A(8), B06(4) -> B11(10), B07(5) -> B12(11)
    terramind_band_mapping = {
        0: 1,  # B02 -> B2
        1: 2,  # B03 -> B3
        2: 3,  # B04 -> B4
        3: 8,  # B05 (NIR) -> B8A
        4: 10, # B06 (SWIR1) -> B11
        5: 11, # B07 (SWIR2) -> B12
    }
# Special handling for Prithvi: map data bands to correct model bands based on spectral content
# Prithvi pretraining uses: B2, B3, B4, B8A, B11, B12
# Data uses: B02, B03, B04, B05, B06, B07 where B05, B06, B07 correspond to B08A, B11, B12
prithvi_band_mapping = None
if args.pangaea_model == 'prithvi':
    # Data bands: B02, B03, B04, B05, B06, B07
    # These correspond spectrally to: Blue, Green, Red, B08A, B11, B12
    # Prithvi expects: B2, B3, B4, B8A, B11, B12
    # Create mapping from data band index to model band index
    # B02(0) -> B2(0), B03(1) -> B3(1), B04(2) -> B4(2), B05(3) -> B8A(3), B06(4) -> B11(4), B07(5) -> B12(5)
    prithvi_band_mapping = {
        0: 0,  # B02 -> B2
        1: 1,  # B03 -> B3
        2: 2,  # B04 -> B4
        3: 3,  # B05 (NIR) -> B8A
        4: 4,  # B06 (SWIR1) -> B11
        5: 5,  # B07 (SWIR2) -> B12
    }

# For Prithvi (single embedding model), use spectral band mapping
# Prithvi uses different bands than what the data band names suggest
if args.pangaea_model and not has_encoder_embeddings:
    if prithvi_band_mapping is not None:
        # Use the spectral mapping for Prithvi
        matching_bands = []
        for data_idx, model_idx in prithvi_band_mapping.items():
            if model_idx < len(model_bands_optical):
                band_name = data_bands_raw[data_idx]
                matching_bands.append({
                    'model_idx': model_idx,
                    'data_idx': data_idx,
                    'band_name': band_name
                })
    else:
        # Fallback: bands already match, use data bands directly
        matching_bands = [{'model_idx': i, 'data_idx': i, 'band_name': data_bands_raw[i]} for i in range(len(data_bands_raw))]
elif terramind_band_mapping is not None:
    # Special case for TerraMind: use the spectral mapping instead of band name matching
    # This ensures correct weight transfer based on actual spectral content
    matching_bands = []
    for data_idx, model_idx in terramind_band_mapping.items():
        if model_idx < len(model_bands_optical):
            band_name = data_bands_raw[data_idx]
            matching_bands.append({
                'model_idx': model_idx,
                'data_idx': data_idx,
                'band_name': band_name
            })
else:
    # Normalize band names for comparison (remove leading zeros: B02 -> B2)
    def normalize_band_name(band_name):
        """Normalize band name by removing leading zeros after 'B'"""
        return re.sub(r'^B0+(\d)', r'B\1', band_name)

    # Create mapping from normalized model band names to indices
    model_bands_normalized = [normalize_band_name(b) for b in model_bands_optical]
    model_band_to_idx = {band: idx for idx, band in enumerate(model_bands_normalized)}

    # Create mapping from normalized data band names to indices
    data_bands_normalized = [normalize_band_name(b) for b in data_bands_raw]
    data_band_to_idx = {band: idx for idx, band in enumerate(data_bands_normalized)}

    # Find matching bands and create mapping
    matching_bands = []
    for data_band in data_bands_normalized:
        if data_band in model_band_to_idx:
            matching_bands.append({
                'model_idx': model_band_to_idx[data_band],
                'data_idx': data_band_to_idx[data_band],
                'band_name': data_band
            })

print(f"Model optical bands: {model_bands_optical}")
print(f"Data bands: {data_bands_raw}")
print(f"Matching bands: {[(m['band_name'], m['model_idx'], m['data_idx']) for m in matching_bands]}")
print(f"Total matching bands: {len(matching_bands)}/{len(data_bands_raw)}")

# Apply selective weight loading for patch embeddings
# Only for pangaea models that have encoder_embeddings
if checkpoint is not None and args.pangaea_model and has_encoder_embeddings:
    for mod_name, emb_module in encoder.encoder_embeddings.items():
        if hasattr(emb_module, 'proj') and hasattr(emb_module.proj, 'weight'):
            checkpoint_proj_key = f"encoder_embeddings.{mod_name}.proj.weight"
            if checkpoint_proj_key in checkpoint:
                pretrained_weights = checkpoint[checkpoint_proj_key]
                
                # Get current model weights shape
                current_weights = emb_module.proj.weight
                
                print(f"Processing {mod_name}: pretrained shape {pretrained_weights.shape}, current shape {current_weights.shape}")
            
            # Determine if this modality is optical or SAR based on num_channels
            # Optical: 12 channels (sen2l2a), SAR: 2 channels (sen1grd)
            num_channels = emb_module.num_channels if hasattr(emb_module, 'num_channels') else 0
            
            # Only process band weights for modalities that have band dimensions
            # For now, only handle optical modalities (12 channels) - SAR will be initialized randomly
            if num_channels == 0 or num_channels > 2:
                # This is an optical modality, process band weights
                # Reshape pretrained weights to separate band dimension
                embed_dim = pretrained_weights.shape[0]
                pixel_dim = pretrained_weights.shape[1]
                
                # Calculate pixel count from patch size (patch_size * patch_size)
                patch_size_val = config['mask']['patch_size']
                pixel_count = patch_size_val * patch_size_val
                
                # Reshape 2D weights [embed_dim, pixel_count * num_bands] to 3D [embed_dim, pixel_count, num_bands]
                if len(pretrained_weights.shape) == 2:
                    # Infer number of pretrained bands from the weight shape
                    num_pretrained_bands_inferred = pixel_dim // pixel_count
                    pretrained_reshaped = pretrained_weights.view(embed_dim, pixel_count, num_pretrained_bands_inferred)
                else:
                    # Already 3D, use as-is
                    pretrained_reshaped = pretrained_weights
                
                num_pretrained_bands = pretrained_reshaped.shape[2] if len(pretrained_reshaped.shape) > 2 else 0
                
                # Create new weights tensor with current model's band count
                num_current_bands = current_weights.shape[2] if len(current_weights.shape) > 2 else 0
                if num_current_bands == 0:
                    new_weights = torch.zeros(embed_dim, pixel_count, len(data_bands_raw), device=current_weights.device, dtype=current_weights.dtype)
                else:
                    new_weights = torch.zeros(embed_dim, pixel_count, num_current_bands, device=current_weights.device, dtype=current_weights.dtype)
                
                # Initialize with random values first (instead of zeros)
                torch.nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
                
                # Copy weights for matching bands
                for match in matching_bands:
                    model_band_idx = match['model_idx']
                    data_band_idx = match['data_idx']
                    
                    if model_band_idx < pretrained_reshaped.shape[2]:
                        new_weights[:, :, data_band_idx] = pretrained_reshaped[:, :, model_band_idx]
                        print(f"  Copied weights for band {match['band_name']}: model_idx={model_band_idx} -> data_idx={data_band_idx}")
                    else:
                        print(f"  Warning: model_band_idx {model_band_idx} out of range for pretrained weights (max: {pretrained_reshaped.shape[2]})")
                
                # Reshape back to 2D [embed_dim, pixel_count * num_bands] before updating model
                new_weights_2d = new_weights.view(embed_dim, -1)
                
                # Update the model's projection weights
                emb_module.proj.weight.data = new_weights_2d
            else:
                # This is a SAR modality with 2 channels, but data doesn't have SAR bands
                # Initialize weights randomly
                print(f"  SAR modality {mod_name} has no matching bands in data, initializing randomly")
                torch.nn.init.kaiming_uniform_(emb_module.proj.weight, a=math.sqrt(5))
            
            # Also update bias if it exists (though projection typically uses bias=False)
            checkpoint_bias_key = f"encoder_embeddings.{mod_name}.proj.bias"
            if checkpoint_bias_key in checkpoint:
                pretrained_bias = checkpoint[checkpoint_bias_key]
                current_bias = emb_module.proj.bias
                
                if current_bias is not None and current_bias.shape[0] == pretrained_bias.shape[0]:
                    emb_module.proj.bias.data = pretrained_bias
                elif current_bias is not None:
                    # Create new bias with current model's band count
                    new_bias = torch.zeros_like(current_bias)
                    torch.nn.init.uniform_(new_bias)
                    
                    # Copy for matching bands
                    for match in matching_bands:
                        model_band_idx = match['model_idx']
                        data_band_idx = match['data_idx']
                        if model_band_idx < pretrained_bias.shape[0]:
                            new_bias[data_band_idx] = pretrained_bias[model_band_idx]
                    
                    emb_module.proj.bias.data = new_bias
            
            print(f"  Updated {mod_name} with {len(matching_bands)} matching bands, {len(data_bands_raw) - len(matching_bands)} bands initialized randomly")
else:
    print("No checkpoint found, skipping selective band weight loading")

if torch.distributed.is_available() and torch.distributed.is_initialized():
    print("init distr")
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank])
    predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[local_rank])

model_has_jepa_target = config['mask']['jepa_target']

# --- Distributed MRR computation ---
encoder.eval()
predictor.eval()

print(f"Interpolate: {args.interpolate}, Model has JEPA target: {model_has_jepa_target}, Pangaea model: {bool(args.pangaea_model)} Therefore: interpolate_not_predict = {args.interpolate or model_has_jepa_target or bool(args.pangaea_model)}")

eval_losses = {"MRR": AverageMeter(), "MRR_var": AverageMeter()}
with torch.no_grad():
    start_time = time.time()
    total_iters = ipve
    for itr, (udata, masks_enc, masks_pred) in enumerate(eval_loader):
        iter_start = time.time()
        print(f'itr {itr}')
        imgs = udata[image_key].to(device, non_blocking=True)
        mrr, mrr_var = metrics.mean_reciprocal_rank(
            encoder,
            predictor,
            imgs,
            patch_size,
            crop_size,
            config['model']['condition_on'],
            device,
            interpolate_not_predict=args.interpolate or model_has_jepa_target or bool(args.pangaea_model),
            pangaea_model=bool(args.pangaea_model),
            sample_mode=args.sample_mode,
            n_aug=n_aug
        )
        eval_losses["MRR"].update(AllReduce.apply(mrr).item())
        eval_losses["MRR_var"].update(AllReduce.apply(mrr_var).item())
        iter_time = (time.time() - iter_start) * 1000  # ms
        avg_time = (time.time() - start_time) / (itr + 1)
        eta = avg_time * (total_iters - itr - 1)
        if torch.distributed.get_rank() == 0:
            print(f"[{itr}] MRR: {eval_losses['MRR'].avg:.4f} (var: {eval_losses['MRR_var'].avg:.4f}) "
                  f"({iter_time:.1f} ms) [eta: {eta:.1f} s]")


print(f"Final MRR: {eval_losses['MRR'].avg:.4f} (var: {eval_losses['MRR_var'].avg:.4f})")
