#!/bin/bash -x
#SBATCH --job-name=finetune_predictor
#SBATCH --account=youraccount
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96          # <--- I DISLIKE THIS, SINCE IT IS HARDWARE-BASED. I'D LIKE A "MAX" SETTING.
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
### ----- ASSERT THAT THE "./slurm_output/" FOLDER EXISTS -----
#SBATCH -o logs/slurm/%j.out # STDOUT
#SBATCH -e logs/slurm/%j.err # STDERR
### ----- END -----
## Mail settings
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=



source venv/activate.sh
# source terramesh/activate.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
# Add dependency continuation:



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

name=iwm_vitb16_hls_256192_predEmb768_corr #! Update this to the name of the model you want to finetune
config_file=logs/${name}/params-ijepa.yaml
checkpoint_file=$(ls "$(dirname "$config_file")"/*50.pth.tar 2>/dev/null | head -n 1)
wandb_name=finetune_${name}
logging_folder=logs/${wandb_name}

data_configs="--logging__folder ${logging_folder} \
        --logging__wandb_name ${wandb_name} \
        --model__finetune_only_predictor True \
        --meta__read_checkpoint $checkpoint_file \
        --resume_preempt \
        --optimization__epochs 301 \
        --optimization__lr 1e-4 \
        --optimization__start_lr 1e-4 \
        --optimization__final_lr 1e-4 \
        --mask__mask_predictions False \
        --mask__mask_context False \
        --data__batch_size 64 \
        "
        # --data__max_images 1000



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
    --fname $config_file \
    --jobid $SLURM_JOB_ID \
    $data_configs \
    "
