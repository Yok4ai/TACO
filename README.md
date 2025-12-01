# TACO RF-DETR Pipeline

A complete pipeline for training RF-DETR (Real-time DEtection TRansformer) on the TACO (Trash Annotations in Context) dataset using PyTorch Lightning.

## Overview

This project provides a comprehensive pipeline for object detection on waste/trash images using the TACO dataset and RF-DETR model. The pipeline includes data preparation, training, and inference scripts with PyTorch Lightning for efficient training and logging.

## Features

- **Complete TACO Dataset Support**: Handles the full TACO dataset with proper train/validation/test splits
- **RF-DETR Integration**: Uses the latest RF-DETR models (Nano, Small, Medium, Base, Large)
- **PyTorch Lightning**: Efficient training with automatic logging, checkpointing, and distributed training support
- **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- **Comprehensive Inference**: Batch inference with visualization and evaluation metrics
- **Easy Configuration**: Environment-based configuration and command-line interfaces

## Dataset

The TACO (Trash Annotations in Context) dataset contains images of litter taken under diverse environments, from tropical beaches to London streets. The dataset includes 60 categories of trash and waste objects.

- **Total Images**: ~1,500 images
- **Categories**: 60 different types of waste/trash
- **Format**: COCO format annotations
- **Splits**: Automatically created train (70%) / validation (20%) / test (10%) splits

## Installation

1. **Clone or download this repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your Roboflow API key (optional):
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

## Usage

### 1. Prepare the Dataset

First, prepare the TACO dataset for training:

```bash
python prepare_taco_dataset.py \
    --data_dir /home/mkultra/Documents/TACO/TACO/data \
    --output_dir /home/mkultra/Documents/TACO/TACO/dataset \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1
```

This script will:
- Create train/validation/test splits
- Copy images to appropriate directories
- Generate separate annotation files for each split
- Create a dataset summary

### 2. Train the Model

Train RF-DETR on the prepared dataset:

```bash
python train_taco_rfdetr.py \
    --dataset_dir /home/mkultra/Documents/TACO/TACO/dataset \
    --model_size medium \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --learning_rate 1e-4 \
    --max_epochs 50 \
    --gpus 1
```

**Training Parameters:**
- `--model_size`: Choose from `nano`, `small`, `medium`, `base`, `large`
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--grad_accum_steps`: Gradient accumulation steps for effective larger batch size
- `--learning_rate`: Learning rate for optimization
- `--max_epochs`: Maximum number of training epochs
- `--gpus`: Number of GPUs to use

**Training Features:**
- Automatic model checkpointing (saves best and last models)
- Early stopping based on validation loss
- TensorBoard and CSV logging
- Learning rate scheduling
- Mixed precision training support

### 3. Run Inference

Perform inference on new images:

**Single Image:**
```bash
python inference_taco_rfdetr.py \
    --input path/to/image.jpg \
    --model_path checkpoints/best_model.ckpt \
    --confidence_threshold 0.5 \
    --show_results
```

**Batch Processing:**
```bash
python inference_taco_rfdetr.py \
    --input path/to/image/directory \
    --output_dir inference_results \
    --model_path checkpoints/best_model.ckpt \
    --save_visualizations \
    --save_json
```

**Evaluation on Test Set:**
```bash
python inference_taco_rfdetr.py \
    --input /home/mkultra/Documents/TACO/TACO/dataset/test \
    --model_path checkpoints/best_model.ckpt \
    --evaluate \
    --test_annotations /home/mkultra/Documents/TACO/TACO/dataset/test/_annotations.coco.json
```

## Project Structure

```
TACO/
├── data/                           # Original TACO dataset
│   ├── annotations.json           # Original annotations
│   ├── batch_1/                   # Image batches
│   ├── batch_2/
│   └── ...
├── dataset/                       # Prepared dataset (created by prepare script)
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg
│   ├── valid/
│   └── test/
├── checkpoints/                   # Model checkpoints
├── logs/                         # Training logs
├── inference_results/            # Inference outputs
├── prepare_taco_dataset.py       # Dataset preparation script
├── taco_rfdetr_lightning.py      # PyTorch Lightning module
├── train_taco_rfdetr.py          # Training script
├── inference_taco_rfdetr.py      # Inference script
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

## Model Sizes and Performance

RF-DETR offers multiple model sizes to balance speed and accuracy:

| Model Size | Parameters | Speed | Accuracy | Use Case |
|------------|------------|-------|----------|----------|
| Nano       | ~3M        | Fastest | Good | Edge devices, real-time |
| Small      | ~10M       | Fast | Better | Mobile applications |
| Medium     | ~25M       | Moderate | High | General purpose |
| Base       | ~50M       | Slower | Higher | High accuracy needs |
| Large      | ~100M      | Slowest | Highest | Research, benchmarking |

## Configuration Options

### Training Configuration

Key training parameters can be adjusted:

- **Batch Size**: Start with 8, reduce if GPU memory issues
- **Learning Rate**: Default 1e-4, can try 1e-3 for faster convergence
- **Image Size**: Default 640x640, can use 512x512 for faster training
- **Epochs**: 50-100 epochs typically sufficient
- **Early Stopping**: Stops training if validation loss doesn't improve

### Data Augmentation

The pipeline includes comprehensive data augmentation:
- Random horizontal flips
- Brightness/contrast adjustments
- Blur effects
- Normalization using ImageNet statistics

### Monitoring and Logging

- **TensorBoard**: Real-time training monitoring
- **CSV Logs**: Structured logging for analysis
- **Model Checkpointing**: Automatic saving of best models
- **Early Stopping**: Prevents overfitting

## Tips for Best Results

1. **GPU Memory**: If you encounter GPU memory issues, reduce batch size and increase gradient accumulation steps
2. **Learning Rate**: Start with 1e-4, reduce if loss oscillates
3. **Data Quality**: Ensure your dataset preparation completed without errors
4. **Monitoring**: Watch TensorBoard logs to track training progress
5. **Inference**: Use appropriate confidence thresholds (0.3-0.7) based on your use case

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Use more GPUs or reduce image size
3. **Poor Performance**: Check data quality, try different learning rates
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Optimization

- Use mixed precision training (`--precision 16`)
- Enable gradient checkpointing for large models
- Use multiple GPUs for faster training
- Optimize data loading with more workers

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this pipeline.

## License

This project is released under the same license as RF-DETR (Apache 2.0). The TACO dataset has its own licensing terms.

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow
- [TACO Dataset](http://tacodataset.org/) by Pedro F. Proença and Pedro Simões
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) for training framework
- [Supervision](https://supervision.roboflow.com/) for computer vision utilities