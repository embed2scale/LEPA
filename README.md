# LEPA

**LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture**

This repository contains the implementation of LEPA, a self-supervised learning method designed specifically for satellite remote sensing imagery. LEPA extends the Image-based Joint-Embedding Predictive Architecture (I-JEPA) framework to learn geometric equivariance properties for patch embeddings.

<details>
<summary>Abstract</summary>

Geospatial foundation models provide precomputed embeddings that serve as compact feature vectors for large-scale satellite remote sensing data. While these embeddings can reduce data-transfer bottlenecks and computational costs, Earth observation (EO) applications can still face geometric mismatches between user-defined areas of interest and the fixed precomputed embedding grid. Standard latent-space interpolation is unreliable in this setting because the embedding manifold is highly non-convex, yielding representations that do not correspond to realistic inputs. We verify this using Prithvi-EO-2.0 to understand the shortcomings of interpolation applied to patch embeddings. As a substitute, we propose a Learned Equivariance-Predicting Architecture (LEPA). Instead of averaging vectors, LEPA conditions a predictor on geometric augmentations to directly predict the transformed embedding. We evaluate LEPA on NASA/USGS Harmonized Landsat-Sentinel (HLS) imagery and ImageNet-1k. Experiments show that standard interpolation achieves a mean reciprocal rank (MRR) below 0.2, whereas LEPA increases MRR to over 0.8, enabling accurate geometric adjustment without re-encoding.

</details>

![LEPA Usage](https://github.com/user-attachments/assets/401ec9aa-eb4f-451f-8908-8d3ed4db14ea)


## Method
LEPA is a method for self-supervised learning that leverages the geometric structure of satellite imagery. The key innovation is learning to predict representations under various geometric transformations (rotations, translations, and scaling) commonly encountered in remote sensing data. This approach:
1. learns representations that are geometrically aware and equivariant to transformations,
3. does not require hand-crafted augmentations or pixel-level reconstruction,
4. enables effective transfer learning for downstream Earth observation tasks.

## Architecture

LEPA consists of:
- **Context Encoder**: Encodes visible patches from satellite imagery
- **Target Encoder**: Processes target regions with exponential moving average (EMA) updates
- **Geometric Predictor**: Predicts target representations conditioned on geometric transformation parameters (rotation angle, scale, translation)

The predictor learns to account for geometric transformations, making the learned representations inherently equivariant and more suitable for satellite remote sensing applications where viewing geometry varies.

## Evaluations

LEPA pretraining learns geometrically-aware representations suitable for Earth observation tasks. The method has been evaluated on the [PANGAEA benchmark](github.com/VMarsocci/pangaea-bench) and achieves competitve results when only using the final embeddings as would be the case in a API setting.

## Launching LEPA pretraining

### Setup on JUWELS
On a compute node run:
```bash
sh venv/setup.sh
source venv/activate.sh
mkdir data
cd data
ln -s /your/path/to/imagenet1k imagenet1k # should contain train/ and val/
```

### Single-GPU training
This implementation starts from the [main.py](main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run JEPA pretraining on 4 GPUs using the config [configs/hls_vitb16.yaml](configs/imnet1k_vitb16.yaml):
```bash
torchrun_jsc \
  --nnodes=1 \
  --nproc_per_node=4 \
  ./src/train.py \
  --fname configs/imnet1k_vitb16.yaml \
  --data__root_path data/imagenet1k/ \
  --evaluation__eval_root_path data/imagenet1k/ \
  --logging__folder logs/jepa_imnet1k \
  --logging__wandb_name jepa_imnet1k \
  --mask__jepa_target True \
  --model__interp_pos_encoding interpolate
```

For hls data instead run
```bash
torchrun_jsc \
  --nnodes=1 \
  --nproc_per_node=4 \
  ./src/train.py \
  --fname configs/hls_vitb16.yaml \
--data__root_path path/to/HLSv9/train \
--evaluation__eval_root_path path/to/HLSv9/val \
--logging__folder logs/jepa_vitb16_hls_224 \
--logging__wandb_name jepa_hls \
--data__input_size 1 224 224 \
--mask__jepa_target True
```

For the full LEPA training run:
```bash
torchrun_jsc \
  --nnodes=1 \
  --nproc_per_node=4 \
  ./src/train.py \
  --fname configs/hls_vitb16.yaml \
--data__root_path path/to/HLSv9/train \
--evaluation__eval_root_path path/to/HLSv9/val \
--logging__folder logs/jepa_vitb16_hls_224 \
--logging__wandb_name jepa_hls \
--data__input_size 1 256 256 \
--data__crop_size 1 192 192
--mask__jepa_target False \
--mask__mask_predictions False \
--model__interp_pos_encoding conditional \
--model__condition_on angle scale tx ty \
```

For more details on the available settings, see the config files in [configs/](configs/) and the argument parser in [src/train.py](src/train.py).


### Multi-GPU training with SLURM
For distributed training on SLURM clusters, use the provided [submit.slurm](submit.slurm) script. The above settings can also be edited in the submission script

### Datasets
This codebase supports:
- **HLS (Harmonized Landsat Sentinel-2)**: Multi-spectral satellite imagery used to pretrain [Prithvi-EO-2.0](https://ieeexplore.ieee.org/abstract/document/11296896)
- **ImageNet-1K**: For baseline comparisons
- Experimental: **TerraMesh**: Earth observation dataset

## Acknowledgments
This work builds upon the [I-JEPA](https://github.com/facebookresearch/ijepa) framework by Assran et al. We thank the authors for open-sourcing their implementation.
This research is carried out as part of the [Embed2Scale](embed2scale.eu) project and is co-funded by the EU Horizon Europe program under Grant Agreement No. 101131841. Additional funding for this project has been provided by the Swiss State Secretariat for Education, Research and Innovation (SERI) and UK Research and Innovation (UKRI).
The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this project by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS at Jülich Supercomputing Centre (JSC).

This project includes code from <a href="https://github.com/facebookresearch/ijepa">I-JEPA</a>.
Licensed under Creative Commons Attribution-NonCommercial 4.0
https://creativecommons.org/licenses/by-nc/4.0/

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation:
```
@misc{scheurer2026lepa,
      title={LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture}, 
      author={Erik Scheurer and Rocco Sedona and Stefan Kesselheim and Gabriele Cavallaro},
      year={2026},
      eprint={2603.07246},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.07246}, 
}
```

### Related Work
```
@misc{assran2023selfsupervisedlearningimagesjointembedding,
      title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture}, 
      author={Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
      year={2023},
      eprint={2301.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.08243}, 
}
```
