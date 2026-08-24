# [CVPR 2026 Highlight] PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning

### [Project Page](https://zju3dv.github.io/PhysSkin/) | [Arxiv](https://arxiv.org/abs/2603.23194) | [Supplementary]()

> [PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning](https://zju3dv.github.io/PhysSkin/),
> Yuanhang Lei, Tao Cheng, Xingxuan Li, Boming Zhao, Siyuan Huang, Ruizhen Hu, Peter Yichen Chen, Hujun Bao, Zhaopeng Cui†

![teaser](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/PhysSkin-CVPR2026/PhysSkin_teaser.png)

PhysSkin learns continuous skinning fields for real-time, physics-based animation across different shapes and discretizations. It combines a transformer-based shape encoder, a cross-attention field decoder, and physics-informed self-supervised objectives.

## Method

![pipeline](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/PhysSkin-CVPR2026/PhysSkin_pipeline.png)

## Installation

The code was tested with Python 3.11, PyTorch 2.7.1, NumPy 1.26.4, PyTorch Lightning 1.6.4, and `conflictfree` 0.1.8.

```bash
python -m pip install "pip<24.1"
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

```text
<dataset_root>/
└── 00000001/
    └── <model_id>/
        └── models/
            └── samples/
                ├── latents.npz
                └── internal_filled.npz
```

- `latents.npz` contains `latents`, a `float32` array of shape `(1, 256, 768)`.
- `internal_filled.npz` contains `points`, a `float32` array of shape `(N, 3)` in approximately `[-0.5, 0.5]`.

The loader rescales the points to approximately `[-1, 1]`. Data splits are provided in `data/splits`, and processed examples for visualization are included in `data/examples`.

## Visualization

```bash
physskin_point_shape_handle_cross_only_vis.ipynb
```

The notebook uses `data/examples` by default. A different sample or dataset can be selected with `MODEL_ID` and `DATASET_ROOT`.

## Training

```bash
DATASET_ROOT=/path/to/dataset \
GPU_IDS="0 1" \
bash scripts/train_physskin.sh
```

## Data preprocessing

```bash
pip install -r data_process/requirements.txt

DATASET_ROOT=/path/to/dataset \
SAMPLING_PYTHON=/path/to/python \
RAYTRACING_PYTHON=/path/to/python \
MICHELANGELO_ROOT=/path/containing/Michelangelo \
ENCODER_CHECKPOINT=/path/to/shape_encoder.pth \
bash data_process/process_data.sh 00000001 0
```

## Checkpoint

The pretrained checkpoint is stored at `checkpoints/physskin_rignet_epoch150.pt`.


## Acknowledgements

This project builds upon [Simplicits](https://github.com/NVIDIAGameWorks/kaolin/tree/master/kaolin/physics/simplicits), [Michelangelo](https://github.com/NeuralCarver/Michelangelo), and [ConFIG](https://github.com/tum-pbs/ConFIG). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
