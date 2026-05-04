# Dual-Resolution Temporal Crop Growth Stage Classification

This repository implements a PyTorch-based framework for crop growth stage classification using dual-resolution temporal remote sensing images. The model takes `20m` (MR) images as the temporal anchor sequence and matches nearby `4m` (HR) images when available. The two image streams are fused with a missing-aware gated module and then modeled with an LSTM for frame-level growth stage prediction.

This repository provides the code implementation and example data for the paper:

## Multi-Scale Spatial-Temporal Remote Sensing Fusion for Phenology Identification in Rice Germplasm Resources

Huimin Wang, Wei Guo, Yue Mu*, Yang Zhang, Haozhou Wang, Yandong Yang, Hu Xu, Yanfeng Ding, Shirong Zhou, Ganghua Li, Seishi Ninomiya

## Method Overview

The model follows this pipeline:

1. Extract image features from the `20m` (MR) stream and the `4m` (HR) stream with two CNN backbones.
2. Project both feature streams into the same embedding dimension.
3. Fuse the two streams with a missing-aware gated fusion module.
4. Feed the fused temporal sequence into an LSTM.
5. Predict the crop growth stage for each time step.

## Repository Structure

```text
.
|-- dataset.py        # Dual-resolution temporal dataset and collate function
|-- doule_train.py    # Training, validation, and testing pipeline
|-- model_mv3.py      # Dual-branch gated temporal fusion model
|-- predict.py        # Batch prediction with trained best.pt checkpoints
|-- utils.py          # Logging and confusion matrix utilities
|-- requirements.txt  # Python dependency list
|-- dataset/  # Prediction dataset
|-- weight/  #  Trained model checkpoints
`-- README.md
```

## Requirements

Recommended environment:

```text
python=3.8
torch==2.4.1+cu118
torchvision==0.19.1
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
pillow==10.4.0
seaborn==0.13.2
```

For GPU acceleration, install the PyTorch build that matches your CUDA version:

```text
https://pytorch.org/get-started/locally/
```

## Training and Prediction

Before training, update the dataset and result paths in `doule_train.py`:

```python
datasetfolder = r"your dataset folder"
resultfolder = r"your result folder"
```

Before full training, it is recommended to run a quick check:

```python
FAST_DEBUG = True
```

`FAST_DEBUG = True` uses only a small subset of the data. It is intended to quickly verify that the dataset paths, class folders, filename date parsing, plot ID parsing, MR-HR image matching, and DataLoader settings are correct. The metrics from this mode are only for debugging and should not be used as final experimental results.

Run training:

```bash
python doule_train.py
```

After confirming that the data can be loaded correctly, set:

```python
FAST_DEBUG = False
```

Then run full training again:

```bash
python doule_train.py
```

Before prediction, update the paths in `predict.py`:

```python
TEST_ROOT_20M = r"\test"
TEST_ROOT_4M  = r"\test"
SAVE_ROOT     = r"\pre_save_root_test"
weight        = r"\weight"
```

`weight` should point to the training output root directory, not to a single `.pt` file. The prediction script automatically looks for the corresponding best checkpoint under:

```text
weight/checkpoints/delta*_mobilenetv3_mobilenetv3_T5_proj512/*_best.pt
```

Run prediction:

```bash
python predict.py
```

## Dataset Format

The training script expects `20m` and `4m` images to be organized separately, with `train`, `val`, and `test` splits. Each split should contain class folders.

Expected directory structure:

```text
your_dataset_folder/
|-- 20m_split_stage_new/
|   |-- train/
|   |   |-- class_1/
|   |   |-- class_2/
|   |   `-- ...
|   |-- val/
|   |   |-- class_1/
|   |   |-- class_2/
|   |   `-- ...
|   `-- test/
|       |-- class_1/
|       |-- class_2/
|       `-- ...
`-- 4m_split_stage_new/
    |-- train/
    |   |-- class_1/
    |   |-- class_2/
    |   `-- ...
    |-- val/
    |   |-- class_1/
    |   |-- class_2/
    |   `-- ...
    `-- test/
        |-- class_1/
        |-- class_2/
        `-- ...
```

### Class Folders

- Class names are inferred from first-level subfolders under each split.
- The `20m` and `4m` sides should use the same class folder names.
- If the two sides are not identical, the code uses their intersection.
- Class indices are assigned by sorted folder names.

### Filename Requirements

The dataset loader parses dates and plot IDs directly from image filenames.

Supported date patterns:

- `YYYYMMDD`
- `YYYYMMDDhhmmss`
- `YYMMDD`

Plot ID parsing:

- The loader first checks the first underscore-separated token.
- It looks for tokens matching `[A-Za-z]+_?\d+`.
- If no such token is found, it falls back to the first token containing letters.

Examples:

```text
JC239_20230415.jpg
plot_12_xxx_20230415.png
AB123_230415.tif
```

Supported image extensions include `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.bmp`.

Dataset and weight is loaded automatically from Hugging Face:

VMAE_MODEL_DIR = "Helen0808/Phenology_Identification"


## Prediction Examples

Prediction examples will be added here.
