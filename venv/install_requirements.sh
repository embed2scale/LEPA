pip install numpy==1.26.4
pip install torch torchrun_jsc
pip install torchvision

pip install accelerate
pip install timm
pip install transformers
pip install torchmetrics
pip install torch_fidelity
pip install diffusers
pip install torchdiffeq
pip install einops
pip install deepspeed
pip install wandb
pip install ninja
pip install flash-attn
pip install omegaconf
pip install matplotlib
pip install beartype
pip install kornia # for magvit2
pip install PyAV
pip install moviepy 
pip install imageio

pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types eg for H100:
export TORCH_CUDA_ARCH_LIST="8.0"
pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.03#egg=xformers

# for 4M dataloading:
# pip install six dateutils datasets xarray albumentations boto3 braceexpand datasets diffusers einops ftfy huggingface opencv-python opencv_python_headless torchmetrics torchrun_jsc tokenizers wandb webdataset zarr # terratorch



