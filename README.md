# TACO RT-DETR v2 Training Pipeline

A complete pipeline for training RT-DETR v2 (Real-time DEtection TRansformer v2) on the TACO (Trash Annotations in Context) dataset using HuggingFace Transformers.

## Overview

This project provides a comprehensive pipeline for object detection on waste/trash images using the TACO dataset and RT-DETR v2 model from HuggingFace. The pipeline supports both COCO and YOLO formats, both using **TACO-10 (10 simplified classes)** for better training accuracy.

**Note:** The original 60 TACO classes don't provide good accuracy in practice. We automatically merge them into 10 simplified categories (TACO-10) for better results.

## Features

- **Dual Format Support**: Choose between COCO format (recommended) or YOLO format
- **TACO-10 Categories**: Automatic conversion from 60 → 10 classes for better accuracy
- **RT-DETR v2 Integration**: Uses HuggingFace's RT-DETR v2 implementation
- **Flexible Augmentation**: Multiple augmentation presets for model soup training strategies
- **COCO Metrics**: Native pycocotools integration for accurate mAP evaluation
- **Easy Pipeline**: Simple scripts for download → prepare → train workflow

## Dataset

The TACO (Trash Annotations in Context) dataset contains images of litter taken under diverse environments, from tropical beaches to London streets.

- **Total Images**: ~1,500 images
- **Original Categories**: 60 different types of waste/trash
- **Training Categories**: 10 TACO-10 simplified categories (merged for better accuracy)
- **Format**: COCO or YOLO format annotations
- **Splits**: Automatically created train (70%) / validation (20%) / test (10%) splits

### TACO-10 Categories

Both COCO and YOLO pipelines use these 10 simplified categories:

1. **Bottle** - All types of bottles (plastic, glass, clear, etc.)
2. **Bottle cap** - Plastic and metal bottle caps
3. **Can** - Food cans, drink cans, aerosols
4. **Cigarette** - Cigarette butts
5. **Cup** - All types of cups (paper, plastic, foam, glass)
6. **Lid** - Plastic and metal lids
7. **Other** - All other waste items
8. **Bag** - All types of bags (plastic, paper, garbage bags)
9. **Pop tab** - Can tabs
10. **Straw** - Plastic and paper straws

### Dataset Format Options

```
Raw TACO Dataset (60 classes)
        ↓
   [prepare_coco_dataset.py]
        ↓
TACO-10 COCO Format (10 classes) ← RECOMMENDED
        │
        └─→ [convert_coco_to_yolo.py] → TACO-10 YOLO Format (10 classes)
```

**Quick Comparison:**
- **COCO Format**: Simpler, faster loading, native RT-DETR support (RECOMMENDED)
- **YOLO Format**: For YOLO toolchain compatibility only

## Installation

### 1. Clone or download this repository

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key Requirements:**
- `torch>=1.11.0` - PyTorch framework
- `transformers>=4.35.0` - HuggingFace Transformers for RT-DETR v2
- `pycocotools` - COCO evaluation metrics
- `albumentations>=1.3.0` - Data augmentation
- `Pillow>=9.0.0` - Image processing
- `scikit-learn>=1.0.0` - Dataset splitting
- `pyyaml` - YOLO config handling

See `requirements.txt` for full list.

## Quick Start Pipeline

### Complete Workflow

```bash
# 1. Download the dataset
python download.py --dataset_path ./data/annotations.json

# 2. Prepare TACO-10 COCO format dataset (60 → 10 classes)
python prepare_coco_dataset.py \
    --data_dir ./data \
    --output_dir ./dataset \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1

# 3. (Optional) Convert to YOLO format if needed
python convert_coco_to_yolo.py

# 4. Train with your chosen format:

# RECOMMENDED: COCO format (simpler and faster)
python train_rtdetrv2_coco.py

# OR: YOLO format (for YOLO compatibility)
python train_rtdetrv2.py
```

## Detailed Usage

### Step 1: Download Dataset

Download TACO images from Flickr using the annotations file:

```bash
python download.py --dataset_path ./data/annotations.json
```

**What it does:**
- Downloads images from Flickr URLs in annotations
- Creates batch subdirectories automatically
- Resumes from where it left off if interrupted
- Preserves EXIF metadata

**Options:**
- `--dataset_path`: Path to TACO annotations.json file (default: `./data/annotations.json`)

### Step 2: Prepare TACO-10 Dataset

Convert the raw TACO dataset (60 classes) into train/val/test splits with TACO-10 (10 classes):

```bash
python prepare_coco_dataset.py \
    --data_dir ./data \
    --output_dir ./dataset \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1
```

**What it does:**
- Merges 60 original classes → 10 TACO-10 classes automatically
- Creates train/validation/test splits
- Copies images to split directories with clean filenames
- Generates `_annotations.coco.json` for each split (TACO-10)
- Creates `dataset_summary.json` with dataset statistics
- Shows category distribution for each split

**Why TACO-10?**
The original 60 classes are too granular and don't provide good accuracy in practice. TACO-10 merges similar categories (e.g., "Clear plastic bottle", "Glass bottle", "Other plastic bottle" → "Bottle") for better classification performance.

**Output Structure:**
```
dataset/
├── train/
│   ├── _annotations.coco.json (TACO-10 format)
│   └── *.jpg (1049 images)
├── valid/
│   ├── _annotations.coco.json (TACO-10 format)
│   └── *.jpg (300 images)
├── test/
│   ├── _annotations.coco.json (TACO-10 format)
│   └── *.jpg (151 images)
└── dataset_summary.json
```

**Options:**
- `--data_dir`: Path to raw TACO data directory
- `--output_dir`: Path for prepared dataset output
- `--train_ratio`: Ratio for training data (default: 0.7)
- `--val_ratio`: Ratio for validation data (default: 0.2)
- `--test_ratio`: Ratio for test data (default: 0.1)

### Step 3: (Optional) Convert to YOLO Format

If you want YOLO format instead of COCO, convert the TACO-10 COCO dataset:

```bash
python convert_coco_to_yolo.py
```

**What it does:**
- Converts COCO format → YOLO format (normalized coordinates)
- Uses TACO-10 categories (same 10 classes as COCO)
- Creates `data.yaml` configuration file
- Organizes into YOLO directory structure

**Output Structure:**
```
yolo_dataset/
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
├── valid/
│   ├── images/*.jpg
│   └── labels/*.txt
├── test/
│   ├── images/*.jpg
│   └── labels/*.txt
└── data.yaml
```

**Note:** Only use this if you specifically need YOLO format. COCO format is simpler and faster.

### Step 4: Train RT-DETR v2

#### Option A: Train with COCO Format (RECOMMENDED)

```bash
python train_rtdetrv2_coco.py
```

**Why COCO format?**
- ✅ Simpler code (no format conversion overhead)
- ✅ Faster data loading
- ✅ Native RT-DETR support
- ✅ Direct COCO metrics evaluation
- ✅ Industry standard format

**Configuration:**

Edit the `CONFIG` dictionary in `train_rtdetrv2_coco.py`:

```python
CONFIG = {
    "model_name": "PekingU/rtdetr_v2_r50vd",  # RT-DETR v2 ResNet-50 backbone
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "img_size": 640,
    "augmentation": "none",  # Change for model soup training
    "run_name": None,  # Auto-generated from augmentation
}
```

**Augmentation Presets:**
- `'none'`: No augmentation (baseline)
- `'flip'`: Horizontal flip only
- `'rotation'`: Rotation ±45°
- `'shear'`: Affine shear ±5°
- `'hsv'`: Color/HSV augmentation
- `'blur'`: Motion/Gaussian/Median blur
- `'noise'`: Gaussian/ISO noise
- `'all'`: All augmentations combined

#### Option B: Train with YOLO Format

```bash
python train_rtdetrv2.py
```

**When to use YOLO format?**
- You specifically need YOLO-format compatibility
- You're comparing with YOLO-based models
- Your deployment pipeline requires YOLO

**Note:** Uses the same augmentation presets and configuration structure as COCO version. Both train on TACO-10 (10 classes).

### Training Features

Both training scripts include:

- **TACO-10 Categories**: Trains on 10 simplified classes
- **COCO Metrics Evaluation**: mAP, mAP@50, mAP@75, Recall
- **Epoch-by-Epoch Tracking**: Metrics logged after each epoch
- **Model Checkpointing**: Saves best model based on validation loss
- **Automatic Logging**: Training logs and metrics saved to JSON
- **GPU Support**: Automatic FP16 mixed precision training
- **Resumable Training**: Can resume from checkpoints

**Training Output:**
```
rtdetr_output/
├── rtdetr_{augmentation}/
│   ├── final_model/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── preprocessor_config.json
│   ├── metrics.json
│   └── checkpoint-{step}/
```

### Model Soup Training Strategy

Train multiple models with different augmentations and ensemble them:

```bash
# Run 1: No augmentation
# Edit CONFIG["augmentation"] = "none"
python train_rtdetrv2_coco.py

# Run 2: Flip augmentation
# Edit CONFIG["augmentation"] = "flip"
python train_rtdetrv2_coco.py

# Run 3: Rotation augmentation
# Edit CONFIG["augmentation"] = "rotation"
python train_rtdetrv2_coco.py

# ... continue with other presets
```

**Ensemble Methods:**
- Weighted averaging of model parameters
- Test-time prediction averaging
- Stacking/voting strategies

Each run saves to: `rtdetr_output/rtdetr_{augmentation}/final_model`

## Project Structure

```
TACO/
├── data/                              # Raw TACO dataset
│   ├── annotations.json              # Original TACO annotations (60 classes)
│   ├── batch_1/, batch_2/, ...       # Downloaded images
│   └── ...
│
├── dataset/                           # TACO-10 COCO format (10 classes)
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg (1049 images)
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg (300 images)
│   ├── test/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg (151 images)
│   └── dataset_summary.json
│
├── yolo_dataset/                      # TACO-10 YOLO format (10 classes, optional)
│   ├── train/
│   │   ├── images/*.jpg
│   │   └── labels/*.txt
│   ├── valid/, test/
│   └── data.yaml
│
├── rtdetr_output/                     # Training outputs
│   └── rtdetr_{augmentation}/
│       ├── final_model/
│       │   ├── model.safetensors
│       │   ├── config.json
│       │   └── preprocessor_config.json
│       ├── metrics.json
│       └── checkpoint-{step}/
│
├── download.py                        # Step 1: Download dataset
├── prepare_coco_dataset.py           # Step 2: Prepare TACO-10 COCO format
├── convert_coco_to_yolo.py           # Step 3: (Optional) Convert to YOLO
├── train_rtdetrv2_coco.py            # Step 4: Train with COCO (recommended)
├── train_rtdetrv2.py                 # Step 4: Train with YOLO (alternative)
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## COCO vs YOLO Format Comparison

Both formats use TACO-10 (10 classes). The only difference is the annotation format.

| Feature | COCO Format | YOLO Format |
|---------|-------------|-------------|
| **Categories** | 10 (TACO-10) | 10 (TACO-10) |
| **Complexity** | Simple | More complex |
| **Code Lines** | ~490 lines | ~660 lines |
| **Loading Speed** | Faster | Slower (conversion overhead) |
| **Bbox Format** | `[x, y, w, h]` absolute | `[x_c, y_c, w, h]` normalized |
| **Conversion Overhead** | None | On-the-fly |
| **RT-DETR Support** | Native | Via conversion |
| **Training Script** | `train_rtdetrv2_coco.py` | `train_rtdetrv2.py` |
| **Use Case** | General purpose, production | YOLO compatibility |
| **Recommended** | ✅ Yes | Only if YOLO needed |

**Recommendation:** Use COCO format (`train_rtdetrv2_coco.py`) unless you specifically need YOLO compatibility. Both produce identical results (same 10 classes), but COCO is simpler and faster.

## Configuration and Hyperparameters

### Model Selection

Available RT-DETR v2 models from HuggingFace:
- `PekingU/rtdetr_v2_r50vd` - ResNet-50 backbone (recommended)
- `PekingU/rtdetr_v2_r101vd` - ResNet-101 backbone (larger)

### Training Hyperparameters

**Key parameters to tune:**

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `batch_size` | 8 | Batch size per GPU | Reduce if OOM, increase if GPU underutilized |
| `learning_rate` | 1e-4 | Initial learning rate | Try 1e-5 for fine-tuning, 1e-3 for faster convergence |
| `epochs` | 50 | Training epochs | 50-100 typical, monitor validation loss |
| `img_size` | 640 | Input image size | 512 for faster, 1024 for better accuracy |
| `warmup_steps` | 500 | LR warmup steps | Increase for larger datasets |

### Data Augmentation Guidelines

**Augmentation Strategy:**

1. **Baseline (none)**: Start here to establish performance ceiling
2. **Light (flip)**: Horizontal flips, minimal overhead
3. **Moderate (rotation, shear)**: Geometric transformations
4. **Heavy (hsv, blur, noise)**: Photometric augmentations
5. **Combined (all)**: Maximum augmentation for robustness

**Model Soup Tip:** Train separate models with different augmentation strategies, then ensemble for best results.

## Kaggle Integration

Both training scripts support Kaggle environments. Upload your chosen dataset format:

**For COCO format (recommended):**
1. Upload `dataset/` folder to Kaggle Datasets as `taco-coco10`
2. Edit `train_rtdetrv2_coco.py`: `DATASET_ROOT = "/kaggle/input/taco-coco10"`
3. Outputs save to: `/kaggle/working/rtdetr_output`

**For YOLO format:**
1. Upload `yolo_dataset/` folder to Kaggle Datasets as `taco-yolo10`
2. Edit `train_rtdetrv2.py`: `DATASET_ROOT = "/kaggle/input/taco-yolo10"`
3. Outputs save to: `/kaggle/working/rtdetr_output`

The scripts automatically detect Kaggle environment (`os.path.exists("/kaggle")`).

## Tips for Best Results

### GPU Memory Optimization

If you encounter CUDA OOM errors:

1. **Reduce batch size**: `CONFIG["batch_size"] = 4`
2. **Use gradient accumulation**: Modify `TrainingArguments.gradient_accumulation_steps`
3. **Reduce image size**: `CONFIG["img_size"] = 512`
4. **Use smaller model**: Try different backbone
5. **Enable gradient checkpointing**: Add to model config

### Training Monitoring

**What to watch:**
- Training loss should decrease steadily
- Validation mAP should increase
- Check for overfitting (train/val gap)
- Monitor GPU utilization (should be >80%)

**Early stopping indicators:**
- Validation loss plateaus for 5+ epochs
- mAP stops improving
- Loss oscillates wildly (reduce LR)

### Performance Optimization

1. **Data Loading**: Increase `dataloader_num_workers` if CPU is idle
2. **Mixed Precision**: Enabled by default with `fp16=True`
3. **Caching**: RT-DETR processor caches preprocessing
4. **Batch Size**: Larger batches = better GPU utilization

## Evaluation and Metrics

### COCO Metrics Explained

Both scripts compute standard COCO metrics on TACO-10 (10 classes):

| Metric | Description | Target |
|--------|-------------|--------|
| `mAP` | Mean Average Precision @ IoU 0.5:0.95 | Primary metric |
| `mAP_50` | mAP @ IoU 0.5 (lenient) | Should be higher |
| `mAP_75` | mAP @ IoU 0.75 (strict) | More precise |
| `mAP_small` | mAP for small objects (<32²) | Often lower |
| `mAP_medium` | mAP for medium objects (32²-96²) | Typically best |
| `mAP_large` | mAP for large objects (>96²) | Usually high |
| `recall` | Detection recall @ IoU 0.5:0.95 | Coverage metric |

**Good performance on TACO-10:**
- mAP: 0.40-0.60 (better than 60 classes)
- mAP_50: 0.60-0.80
- Recall: 0.50-0.70

**Note:** TACO-10 typically achieves 10-15% higher mAP than training on all 60 classes due to reduced class confusion.

### Interpreting Results

**Metrics saved to:** `rtdetr_output/rtdetr_{augmentation}/metrics.json`

```json
{
  "final_metrics": {
    "mAP": 0.5234,
    "mAP_50": 0.7123,
    "mAP_75": 0.5567,
    "recall": 0.6234
  },
  "epoch_metrics": [
    {"epoch": 1, "mAP": 0.2134, ...},
    {"epoch": 2, "mAP": 0.3245, ...},
    ...
  ]
}
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
→ Reduce `batch_size`, reduce `img_size`, or use gradient accumulation

**2. Slow Training**
```
Training takes too long
```
→ Increase `dataloader_num_workers`, reduce `img_size`, or use smaller dataset

**3. Import Errors**
```
ModuleNotFoundError: No module named 'transformers'
```
→ Reinstall requirements: `pip install -r requirements.txt`

**4. Poor Performance**
```
mAP stays very low
```
→ Check data quality, increase training epochs, try different augmentation, verify annotations

**5. Dataset Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../dataset/train'
```
→ Run `prepare_coco_dataset.py` first to create dataset structure

### Getting Help

**Before asking for help:**
1. Check error message carefully
2. Verify dataset preparation completed successfully
3. Confirm GPU is available: `torch.cuda.is_available()`
4. Check disk space and permissions
5. Review configuration values

**Debug mode:**
```python
# Add to training script for more verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Benchmarks

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (e.g., GTX 1080, RTX 2070)
- RAM: 16GB
- Disk: 10GB free space

**Recommended:**
- GPU: 16GB+ VRAM (e.g., RTX 3090, V100)
- RAM: 32GB
- Disk: 20GB+ free space

### Training Time Estimates

| Configuration | GPU | Batch Size | Time per Epoch | Total Time (50 epochs) |
|--------------|-----|------------|----------------|----------------------|
| Default | RTX 3090 | 8 | ~3 min | ~2.5 hours |
| Default | V100 | 8 | ~4 min | ~3.5 hours |
| Large batch | A100 | 16 | ~2 min | ~1.5 hours |
| Small GPU | GTX 1080 | 4 | ~6 min | ~5 hours |

*Times are approximate for TACO-10 dataset (~1000 training images)*

## Contributing

Contributions are welcome! Areas for improvement:
- Additional augmentation strategies
- Model ensemble implementation
- Inference/deployment scripts
- Performance optimizations
- Documentation improvements

## License

This project uses:
- RT-DETR v2 model: Apache 2.0 License
- TACO Dataset: See [TACO website](http://tacodataset.org/) for licensing
- Code: MIT License (or specify your preferred license)

## Acknowledgments

- [RT-DETR v2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) by Peking University
- [TACO Dataset](http://tacodataset.org/) by Pedro F. Proença and Pedro Simões
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) for model implementation
- [pycocotools](https://github.com/cocodataset/cocoapi) for evaluation metrics
- [Albumentations](https://albumentations.ai/) for data augmentation

## Citation

If you use this pipeline or TACO dataset in your research, please cite:

```bibtex
@article{taco2019,
  title={TACO: Trash Annotations in Context Dataset},
  author={Proença, Pedro F and Simões, Pedro},
  journal={arXiv preprint arXiv:2003.06975},
  year={2020}
}

@article{rtdetr,
  title={RT-DETR: DETRs Beat YOLOs on Real-time Object Detection},
  author={Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  journal={arXiv preprint arXiv:2304.08069},
  year={2023}
}
```

## Additional Resources

- **TACO Dataset**: http://tacodataset.org/
- **RT-DETR Paper**: https://arxiv.org/abs/2304.08069
- **HuggingFace RT-DETR Docs**: https://huggingface.co/docs/transformers/model_doc/rt_detr
- **COCO Format Specification**: https://cocodataset.org/#format-data
- **Albumentations Docs**: https://albumentations.ai/docs/

---

**Questions or issues?** Please open an issue on the repository or refer to the troubleshooting section above.
