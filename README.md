# MRAD: Memory-based Retrieval Augmented Detection

> **MRAD: Memory-based Retrieval Augmented Detection for Zero-shot Anomaly Detection**

## Introduction

MRAD is a novel approach for zero-shot anomaly detection that leverages memory banks and CLIP features. By building retrieval-augmented memory banks from auxiliary datasets, MRAD achieves strong cross-domain anomaly detection and localization without any target domain training data. This repository provides three model variants with different trade-offs:

![Overview of MRAD](./figures/mrad.png)

| Model | Description | Training | Weights |
|-------|------------|----------|---------|
| **MRAD-CLIP** | Full model with memory bank + CLIP image-text alignment | 6 epochs | patch_proj, image_proj, prompt_learner, prompt_proj |
| **MRAD-FT** | Fine-tuned model with memory bank projection | 1 epoch | patch_proj, image_proj |
| **MRAD-TF** | Train-free baseline using raw CLIP features | None | None |



### Install dependencies --- Python 3.9, PyTorch 2.0.0+cu117

```bash
pip install -r requirements.txt
```

### Prepare your dataset

Download the dataset below:

* Industrial Domain:
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip),  [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)

* Medical Domain:
[HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage), [BrainMRI](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [ISIC](https://drive.google.com/file/d/1UeuKgF1QYfT1jTlYHjxKB3tRjrFHfFDR/view), [Endo](https://drive.google.com/file/d/1LNpLkv5ZlEUzr_RPN5rdOHaqk0SkZa3m/view)

### Generate the dataset JSON (same as [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP/tree/main?tab=readme-ov-file))

Take MVTec AD for example:

Structure of MVTec Folder:
```
mvtec/
│
├── meta.json
│
├── bottle/
│   ├── ground_truth/
│   │   ├── broken_large/
│   │   │   └── 000_mask.png
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── broken_large/
│       │   └── 000.png
│       │   └── ...
│       └── ...
│
└── ...
```

```bash
cd generate_dataset_json
python mvtec.py
```

Select the corresponding script and run it. The generated JSON stores all the information that MRAD needs.


### Run MRAD

* Quick start with pre-trained weights (MRAD-CLIP, trained on VisA, tested on MVTec):

```bash
python test.py \
    --model_type mrad-clip \
    --checkpoint_path ./checkpoints/test_on_mvtec.pth \
    --dataset mvtec \
    --data_path ./data/MVTEC \
    --cache_dir ./cache \
    --save_path ./results/test_on_mvtec \
    --device cuda:0
```

* Train your own weights:

```bash
python train.py \
    --model_type mrad-clip \
    --dataset visa \
    --data_path ./data/VISA \
    --save_path ./checkpoints/mrad_clip \
    --cache_dir ./cache \
    --device cuda:0
```

Before running the above commands, please download the official memory banks from [this link](https://drive.google.com/drive/folders/1XLl297B_jnu9Qv5KPZEw8MeAY9JJnaZg?usp=drive_link) and the corresponding MRAD-CLIP model weights from [this link](https://drive.google.com/drive/folders/11lugxXdal5_3RUhr9UcC2JItp95C4aIv?usp=drive_link), and place them under `./cache` and `./checkpoints` respectively.

The `./cache` directory stores the memory banks built from MVTec and VisA. If you want to rebuild the memory banks from scratch, please set the `load_cache` argument of both `build_cache_model` and `build_patch_cache_model` to `False` in `train.py`.

* End-to-end pipeline (train on VisA + test on MVTec):

If you want to retrain from scratch, you can use the following command as a quick start, but please set the GPU device according to your actual environment.

```bash
bash run.sh mrad-clip
```

You can switch model variants by changing `--model_type` to `mrad-ft` or `mrad-clip`.

## Experiment setting

We train on VisA and test on all other datasets. For VisA, MRAD is trained on MVTec.


## Acknowledgments

We thank the authors of the following code repositories: [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), [open_clip](https://github.com/mlfoundations/open_clip), [CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery).


## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```bibtex
@inproceedings{
xu2026mrad,
title={{MRAD}: Zero-Shot Anomaly Detection with Memory-Driven Retrieval},
author={Chaoran Xu and Chengkan Lv and Qiyu Chen and Feng Zhang and Zhengtao Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
```
