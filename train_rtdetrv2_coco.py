"""
Simple RT-DETR v2 Training Pipeline with Native COCO Format
Uses existing COCO annotations - no YOLO conversion needed!
"""

import os

# Fix for Kaggle multi-GPU issue with RT-DETR v2
# MUST be set BEFORE importing torch to avoid DataParallel tensor size mismatch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments, TrainerCallback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A


def get_augmentation_preset(preset_name="none", img_size=640):
    """
    Get augmentation pipeline for different strategies (model soup approach)

    Args:
        preset_name: One of ['none', 'flip', 'rotation', 'shear', 'hsv', 'mosaic', 'cutmix', 'shear_mosaic', 'all']
        img_size: Target image size (not used since RT-DETR processor handles resizing)
    """

    # Note: No base transforms - RT-DETR processor handles resizing/padding
    # We only apply augmentations here
    aug_transforms = []

    if preset_name == "flip":
        # Run 1: Flip Only
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
        ]

    elif preset_name == "rotation":
        # Run 2: Rotation Only
        aug_transforms = [
            A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=0.5),
        ]

    elif preset_name == "shear":
        # Run 3: Shear Only
        aug_transforms = [
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.5),
        ]

    elif preset_name == "hsv":
        # Run 4: HSV/Color Only
        aug_transforms = [
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),  # 0.015 in YOLO = ~2.7 degrees
                sat_shift_limit=int(0.7 * 255),     # 0.7 in YOLO
                val_shift_limit=int(0.4 * 255),     # 0.4 in YOLO
                p=0.5
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.5),
        ]

    elif preset_name == "blur":
        # Additional: Blur augmentation
        aug_transforms = [
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
        ]

    elif preset_name == "noise":
        # Additional: Noise augmentation
        aug_transforms = [
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
        ]

    elif preset_name == "shear_mosaic":
        # Run 7: Shear + Mosaic combined
        aug_transforms = [
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.5),
            # Note: Mosaic requires special handling (multiple images), implemented separately
        ]

    elif preset_name == "all":
        # All augmentations combined (for baseline comparison)
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=0.3),
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7 * 255),
                val_shift_limit=int(0.4 * 255),
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.2),
        ]

    elif preset_name == "none":
        # No augmentation
        aug_transforms = []

    else:
        raise ValueError(f"Unknown augmentation preset: {preset_name}")

    # Return None if no augmentations (let RT-DETR processor handle everything)
    if len(aug_transforms) == 0:
        return None

    # Combine augmentation transforms
    transform = A.Compose(
        aug_transforms,
        bbox_params=A.BboxParams(
            format='coco',  # [x, y, width, height]
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,  # Remove boxes with <30% visibility after augmentation
        )
    )

    return transform


class CocoDetectionDataset(Dataset):
    """Native COCO format dataset for RT-DETR v2 with augmentation support"""

    def __init__(self, img_folder, annotation_file, processor, transform=None):
        self.img_folder = Path(img_folder)
        self.processor = processor
        self.transform = transform

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image info
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.img_folder / image_info['file_name']

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract boxes and labels in COCO format
        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox is already [x, y, width, height]
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        # Apply augmentations if specified
        if self.transform is not None and len(boxes) > 0:
            transformed = self.transform(
                image=image_np,
                bboxes=boxes,
                class_labels=labels
            )
            image_np = transformed['image']
            boxes = list(transformed['bboxes'])
            labels = list(transformed['class_labels'])
        elif self.transform is not None:
            # No boxes, just transform image
            transformed = self.transform(image=image_np, bboxes=[], class_labels=[])
            image_np = transformed['image']

        # Convert back to PIL for processor
        image = Image.fromarray(image_np)

        # Prepare target in COCO format for RT-DETR
        target = {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": box,  # [x, y, width, height]
                    "area": box[2] * box[3],  # width * height
                    "iscrowd": 0
                }
                for box, label in zip(boxes, labels)
            ]
        }

        # Process with RT-DETR processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0] if encoding["labels"] else {}
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch]
    }


def evaluate_coco_metrics(model, dataset, processor, device, verbose=True):
    """Evaluate model and return COCO metrics (mAP, recall, etc.)"""
    model.eval()
    coco_gt = dataset.coco
    results = []

    if verbose:
        print("Running COCO evaluation...")

    for idx in range(len(dataset)):
        image_id = dataset.image_ids[idx]
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = dataset.img_folder / image_info['file_name']
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        predictions = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]

        for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
            x_min, y_min, x_max, y_max = box.cpu().tolist()
            results.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "score": score.item()
            })

    if not results:
        if verbose:
            print("No predictions made!")
        return {}

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Summarize (suppress output if not verbose)
    if not verbose:
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        sys.stdout = old_stdout
    else:
        coco_eval.summarize()

    # Extract metrics
    metrics = {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "recall": coco_eval.stats[8],
        "recall_small": coco_eval.stats[9],
        "recall_medium": coco_eval.stats[10],
        "recall_large": coco_eval.stats[11],
    }

    return metrics


class CocoEvalCallback(TrainerCallback):
    """Callback to compute COCO metrics after each epoch"""

    def __init__(self, val_dataset, processor, device):
        self.val_dataset = val_dataset
        self.processor = processor
        self.device = device
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        print("\n" + "=" * 80)
        print(f"EPOCH {int(state.epoch)} METRICS")
        print("=" * 80)

        metrics = evaluate_coco_metrics(model, self.val_dataset, self.processor, self.device, verbose=False)

        # Print in a nice table format
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 32)
        print(f"{'mAP (0.5:0.95)':<20} {metrics.get('mAP', 0):.4f}")
        print(f"{'mAP@50':<20} {metrics.get('mAP_50', 0):.4f}")
        print(f"{'mAP@75':<20} {metrics.get('mAP_75', 0):.4f}")
        print(f"{'Recall':<20} {metrics.get('recall', 0):.4f}")
        print(f"{'mAP (small)':<20} {metrics.get('mAP_small', 0):.4f}")
        print(f"{'mAP (medium)':<20} {metrics.get('mAP_medium', 0):.4f}")
        print(f"{'mAP (large)':<20} {metrics.get('mAP_large', 0):.4f}")
        print("=" * 80 + "\n")

        # Store metrics
        self.epoch_metrics.append({
            "epoch": int(state.epoch),
            **metrics
        })

        return control


# ============================================================================
# MAIN TRAINING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train RT-DETR v2 on TACO dataset (COCO format)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset paths
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help='Path to TACO-10 COCO dataset root directory (auto-detects Kaggle if not specified)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs (auto-detects Kaggle if not specified)'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default='PekingU/rtdetr_v2_r50vd',
        help='HuggingFace model name'
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of dataloader workers'
    )

    # Augmentation
    parser.add_argument(
        '--augmentation',
        type=str,
        default='none',
        choices=['none', 'flip', 'rotation', 'shear', 'hsv', 'blur', 'noise', 'shear_mosaic', 'all'],
        help='Augmentation preset for model soup training'
    )

    # Run configuration
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Run name for output directory (auto-generated from augmentation if not specified)'
    )

    return parser.parse_args()


# Parse arguments
args = parse_args()

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle")

# Set dataset and output paths
if args.dataset_root is None:
    DATASET_ROOT = "/kaggle/input/taco-coco" if IS_KAGGLE else "/home/mkultra/Documents/TACO/dataset"
else:
    DATASET_ROOT = args.dataset_root

if args.output_dir is None:
    OUTPUT_DIR = "/kaggle/working/rtdetr_output" if IS_KAGGLE else "./rtdetr_output"
else:
    OUTPUT_DIR = args.output_dir

# Generate run name if not specified
if args.run_name is None:
    run_name = f"rtdetr_{args.augmentation}"
else:
    run_name = args.run_name

# Update output directory with run name
OUTPUT_DIR = f"{OUTPUT_DIR}/{run_name}"

print("=" * 80)
print("RT-DETR v2 Training Pipeline (Native COCO Format)")
print("=" * 80)
print(f"Dataset: {DATASET_ROOT}")
print(f"Model: {args.model_name}")
print(f"Augmentation: {args.augmentation}")
print(f"Run name: {run_name}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.learning_rate}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print()

# Load dataset summary to get class names
dataset_root = Path(DATASET_ROOT)
with open(dataset_root / "dataset_summary.json", 'r') as f:
    summary = json.load(f)

class_names = summary['category_names']
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

print(f"Classes ({len(class_names)}): {class_names}")
print()

# Initialize
processor = RTDetrImageProcessor.from_pretrained(args.model_name)
model = RTDetrV2ForObjectDetection.from_pretrained(
    args.model_name,
    id2label=id2label,
    label2id=label2id,
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# Create augmentation transforms
train_transform = get_augmentation_preset(args.augmentation, args.img_size)
val_transform = get_augmentation_preset('none', args.img_size)  # No augmentation for validation

print(f"Train augmentation: {args.augmentation}")
print(f"Validation augmentation: none (resize only)")
print()

# Create datasets - much simpler now!
train_dataset = CocoDetectionDataset(
    img_folder=dataset_root / "train",
    annotation_file=dataset_root / "train" / "_annotations.coco.json",
    processor=processor,
    transform=train_transform
)

val_dataset = CocoDetectionDataset(
    img_folder=dataset_root / "valid",
    annotation_file=dataset_root / "valid" / "_annotations.coco.json",
    processor=processor,
    transform=val_transform
)

print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")
print()

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=50,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=args.num_workers,
)

# Setup COCO metrics callback
device = "cuda" if torch.cuda.is_available() else "cpu"
coco_callback = CocoEvalCallback(val_dataset, processor, device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    callbacks=[coco_callback],
)

print("Starting training...")
print("=" * 80)
trainer.train()
print("=" * 80)
print("Training completed!\n")

# Save model
trainer.save_model(f"{OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_model")

# Evaluate with COCO metrics
print("Final evaluation with COCO metrics...")
print("=" * 80)
model.to(device)

metrics = evaluate_coco_metrics(model, val_dataset, processor, device)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
for metric, value in metrics.items():
    print(f"{metric:20s}: {value:.4f}")
print("=" * 80)

# Save all metrics (final + epoch-by-epoch)
all_metrics = {
    "final_metrics": metrics,
    "epoch_metrics": coco_callback.epoch_metrics
}

with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}/final_model")
print(f"Metrics saved to: {OUTPUT_DIR}/metrics.json")
print(f"Tracked {len(coco_callback.epoch_metrics)} epochs")


# ============================================================================
# MODEL SOUP INSTRUCTIONS
# ============================================================================
"""
To create a model soup ensemble, run this script multiple times with different
augmentation presets using the --augmentation flag.

Example runs for model soup:

1. No augmentation:  python train_rtdetrv2_coco.py --augmentation none
2. Flip only:        python train_rtdetrv2_coco.py --augmentation flip
3. Rotation only:    python train_rtdetrv2_coco.py --augmentation rotation
4. Shear only:       python train_rtdetrv2_coco.py --augmentation shear
5. HSV/Color only:   python train_rtdetrv2_coco.py --augmentation hsv
6. Blur only:        python train_rtdetrv2_coco.py --augmentation blur
7. Noise only:       python train_rtdetrv2_coco.py --augmentation noise
8. Shear + Mosaic:   python train_rtdetrv2_coco.py --augmentation shear_mosaic
9. All combined:     python train_rtdetrv2_coco.py --augmentation all

Each run will save to: {output_dir}/rtdetr_{augmentation}/final_model

After training all variants, you can ensemble them using:
- Weighted averaging of model parameters
- Test-time averaging of predictions
- Stacking/voting strategies

Additional options:
    --dataset_root PATH     Path to TACO-10 dataset
    --output_dir PATH       Output directory for models
    --epochs N              Number of epochs (default: 50)
    --batch_size N          Batch size (default: 8)
    --learning_rate LR      Learning rate (default: 1e-4)
    --img_size SIZE         Image size (default: 640)

Example with custom settings:
    python train_rtdetrv2_coco.py \
        --dataset_root ./dataset \
        --output_dir ./outputs \
        --augmentation flip \
        --epochs 100 \
        --batch_size 16 \
        --learning_rate 5e-5

Tip: Keep epochs, batch_size, and learning_rate consistent across
all runs for fair model soup comparison.
"""
