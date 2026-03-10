# Compute the Mean Reciprocal Rank (MRR) on the entire eval dataset using distributed data parallel (torchrun).
# This cell assumes the notebook is run with torchrun and the config path is provided as a command-line argument.


# run with torchrun --nnodes=1 --nproc_per_node=4 ./eval_MRR_fulldataset.py --config logs/iwm_vitb16_hls_256192_cls_corrposcond_correctedcls_nolosschange/params-ijepa.yaml
# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 50 --config logs/jepa_vitb_imagenet_224_register/params-ijepa.yaml > logs/jepa_vitb_imagenet_224_register/eval_MRR.log


# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 55  --sample_mode nearest --config logs/finetune_iwm_vitb16_hls_256192_predEmb768_corr/params-ijepa.yaml > logs/finetune_iwm_vitb16_hls_256192_predEmb768_corr/eval_MRR_nearest.log
# srun --account=youraccount --partition=booster --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --threads-per-core=2 --gres=gpu:4 --time=02:00:00 --pty torchrun --nproc_per_node=4 ./eval_MRR_fulldataset.py --epoch 55  --sample_mode nearest --config logs/finetune_iwm_vitb16_hls_256192_predEmb192_corr/params-ijepa.yaml > logs/finetune_iwm_vitb16_hls_256192_predEmb192_corr/eval_MRR_nearest.log


import torch
import os
import yaml

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
    model_config['input_size']=256

    encoder: Encoder = instantiate(model_config)
    encoder.eval()

    
    config = {'mask':{},'data':{},'evaluation':{},'model':{}}
    config['data']['input_size'] = 256
    config['data']['crop_size'] = 192
    config['mask']['patch_size'] = model_config["patch_size"]
    config['data']['batch_size'] = 128
    config['data']['root_path'] = "data/HLSv9/train"
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
    config['evaluation']['eval_root_path'] = "/p/scratch/youraccount/HLSv9/val"
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

epoch = args.epoch
n_aug = args.n_aug
# Distributed setup
world_size,local_rank,device=init_distributed()
# torch.cuda.set_device(local_rank)


crop_size = 192 #config['data']['crop_size'][-1] if isinstance(config['data']['crop_size'], list) else config['data']['crop_size']
patch_size = config['mask']['patch_size']
input_size = 256 #config['data']['input_size'][-1] if isinstance(config['data']['input_size'], list) else config['data']['input_size']
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
    root_path='/p/scratch/youraccount/HLSv9/train',#config['data']['root_path'],
    eval_root_path='/p/scratch/youraccount/HLSv9/val',#config['evaluation']['eval_root_path'],
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
    )
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

if torch.distributed.is_available() and torch.distributed.is_initialized():
    print("init distr")
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank])
    predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[local_rank])

model_has_jepa_target = config['mask']['jepa_target']

# --- Distributed MRR computation ---
encoder.eval()
predictor.eval()


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
            interpolate_not_predict=model_has_jepa_target,
            pangaea_model=bool(args.pangaea_model),
            sample_mode=args.sample_mode
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
