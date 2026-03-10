#!/bin/bash -x
#SBATCH --job-name=train_ijepa
#SBATCH --account=youraccount
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96          # <--- I DISLIKE THIS, SINCE IT IS HARDWARE-BASED. I'D LIKE A "MAX" SETTING.
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
### ----- ASSERT THAT THE "./slurm_output/" FOLDER EXISTS -----
#SBATCH -o logs/slurm/hlsimnet-%j.out # STDOUT
#SBATCH -e logs/slurm/hlsimnet-%j.err # STDERR
### ----- END -----
## Mail settings
## SBATCH --mail-type=END,FAIL
## SBATCH --mail-user=



source venv/activate.sh
export PYTHONPATH=$(pwd):$PYTHONPATH


# # ----- "Without this, srun does not inherit cpus-per-task from sbatch" -----
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
echo "SRUN_CPUS_PER_TASK: ""$SRUN_CPUS_PER_TASK"

MASTER_ADDRi="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# # Allow communication over InfiniBand cells.
MASTER_ADDRi="${MASTER_ADDRi}i"
# # Get IP for hostname.
export MASTER_ADDRi="$(nslookup "$MASTER_ADDRi" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=33523

export GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

ulimit -c 0 # <--- Disable core dumps
echo "MASTER_ADDR[i]: ""$MASTER_ADDRi"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## CONFIG

# HLS Dataset
## JEPA
# data_configs="--data__root_path data/HLSv9/train \
#         --evaluation__eval_root_path data/HLSv9/val \
#         --logging__folder logs/jepa_vitb16_hls_224 \
#         --logging__wandb_name jepa_hls \
#         --data__input_size 1 224 224 \
#         --model__model_name_enc vit_base \
#         --model__enc_has_cls_token False \
#         --mask__patch_size 16 \
#         --mask__jepa_target True \
#         --mask__allow_overlap False \
#         --mask__collator_type multiblock \
#         --model__interp_pos_encoding interpolate \
#         --data__batch_size 128 \
#         --data__num_workers 8 \
#         "
        # --model__enc_n_register_tokens 8 \
        # --model__pred_n_register_tokens 8 \
        # --resume_preempt \
        # --mask__enc_mask_scale 0.3 0.5 \
        # --mask__pred_mask_scale 0.4 0.6 \
        # --data__max_images 2101504 \

## IWM

data_configs="--data__root_path data/HLSv9/train \
        --evaluation__eval_root_path data/HLSv9/val \
        --logging__folder logs/iwm_vitb16_hls_256192_predEmb192_corr \
        --logging__wandb_name iwm_vitb16_hls_256192_predEmb192_corr \
        --data__input_size 1 256 256 \
        --data__crop_size 1 192 192 \
        --model__model_name_enc vit_base \
        --model__enc_has_cls_token False \
        --mask__patch_size 16 \
        --mask__jepa_target False \
        --mask__allow_overlap False \
        --mask__collator_type multiblock \
        --mask__mask_predictions False \
        --model__interp_pos_encoding conditional \
        --data__batch_size 128 \
        --data__num_workers 8 \
        --model__condition_on angle scale tx ty \
        --model__pred_emb_dim 192 \
        "
        # --resume_preempt \
        # --model__pred_depth 14 \
        # --mask__mask_context False \
        # --mask__enc_mask_scale 0.3 0.5 \
        # --model__enc_n_register_tokens 8 \
        # --model__pred_n_register_tokens 8 \
        # --mask__pred_mask_scale 0.4 0.6 \
        # --data__max_images 2101504 \

# ImageNet1k Dataset
## JEPA
# data_configs="--data__root_path data/imagenet1k/ \
#         --evaluation__eval_root_path data/imagenet1k/ \
#         --data__num_workers 8 \
#         --logging__folder logs/jepa_vitb_imagenet_224_cls_register_corr_nolosschange \
#         --logging__wandb_name jepa_imagenet \
#         --data__bands 1 2 3 \
#         --data__input_size 1 224 224 \
#         --data__batch_size 128 \
#         --mask__patch_size 16 \
#         --mask__jepa_target True \
#         --model__model_name_enc vit_base \
#         --model__interp_pos_encoding interpolate \
#         --mask__allow_overlap False \
#         --model__enc_has_cls_token True \
#         --resume_preempt \
#         --model__enc_n_register_tokens 8 \
#         --model__pred_n_register_tokens 8 \
#         "


write_tag="vitb"

echo "Starting job with the following config:"
echo $data_configs

export OMP_NUM_THREADS=12 # set to number of physical cores per task ie 48/4=12
export WANDB__SERVICE_WAIT=300
srun -l bash -c "torchrun_jsc \
    --nproc_per_node=$GPUS_PER_NODE\
    --nnodes=$SLURM_JOB_NUM_NODES\
    --rdzv_id=$SLURM_JOB_ID\
    --rdzv_backend=c10d\
    --rdzv_endpoint=$MASTER_ADDRi:$MASTER_PORT \
    --rdzv_conf=is_host=\"\$(if((SLURM_PROCID)); then echo 0; else echo 1; fi)\" \
    --max-restarts=0 \
    ./src/train.py \
    --fname configs/hls_vitb16.yaml \
    --jobid $SLURM_JOB_ID \
    $data_configs \
    --logging__write_tag $write_tag \
    "
