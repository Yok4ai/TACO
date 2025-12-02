"""
Simple RT-DETR v2 Training Pipeline with COCO Metrics
Run on Kaggle for TACO Dataset (YOLO Format)
"""

import os

# Fix for Kaggle multi-GPU issue with RT-DETR v2
# MUST be set BEFORE importing torch to avoid DataParallel tensor size mismatch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import yaml
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments, TrainerCallback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YoloDetectionDataset(Dataset):
    """YOLO format dataset for RT-DETR v2"""

    def __init__(self, img_folder, label_folder, processor, class_names):
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder)
        self.processor = processor
        self.class_names = class_names

        # Get all image files
        self.image_files = sorted(list(self.img_folder.glob("*.jpg")) + list(self.img_folder.glob("*.png")))

    def __len__(self):
        return len(self.image_files)

    def yolo_to_coco(self, yolo_bbox, img_width, img_height):
        """Convert YOLO format (x_center, y_center, w, h) normalized to COCO format [x, y, width, height] absolute"""
        x_center, y_center, width, height = yolo_bbox

        # Convert to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Convert to COCO format (top-left corner x, y, width, height)
        x = x_center - width / 2
        y = y_center - height / 2

        return [x, y, width, height]

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Load corresponding label file
        label_path = self.label_folder / (img_path.stem + ".txt")

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        yolo_bbox = [float(x) for x in parts[1:5]]

                        # Convert YOLO bbox to COCO format [x, y, width, height]
                        coco_bbox = self.yolo_to_coco(yolo_bbox, img_width, img_height)

                        boxes.append(coco_bbox)
                        labels.append(class_id)

        # Prepare target in COCO format
        target = {
            "image_id": idx,
            "annotations": [
                {
                    "image_id": idx,
                    "category_id": label,
                    "bbox": box,  # [x, y, width, height]
                    "area": box[2] * box[3],  # width * height
                    "iscrowd": 0
                }
                for box, label in zip(boxes, labels)
            ]
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0] if encoding["labels"] else {}
        }


class YoloToCoco:
    """Convert YOLO dataset to COCO format for evaluation"""

    def __init__(self, img_folder, label_folder, class_names):
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder)
        self.class_names = class_names
        self.image_files = sorted(list(self.img_folder.glob("*.jpg")) + list(self.img_folder.glob("*.png")))

        # Build COCO format annotations
        self.coco_data = self._build_coco()
        self.coco = COCO()
        self.coco.dataset = self.coco_data
        self.coco.createIndex()

    def _build_coco(self):
        """Build COCO format dictionary from YOLO dataset"""
        images = []
        annotations = []
        ann_id = 0

        for img_id, img_path in enumerate(self.image_files):
            img = Image.open(img_path)
            img_width, img_height = img.size

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_width,
                "height": img_height
            })

            # Load annotations
            label_path = self.label_folder / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = [float(x) for x in parts[1:5]]

                            # Convert to absolute COCO format [x, y, width, height]
                            x_center *= img_width
                            y_center *= img_height
                            width *= img_width
                            height *= img_height

                            x = x_center - width / 2
                            y = y_center - height / 2

                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": class_id,
                                "bbox": [x, y, width, height],
                                "area": width * height,
                                "iscrowd": 0
                            })
                            ann_id += 1

        categories = [{"id": i, "name": name, "supercategory": name} for i, name in enumerate(self.class_names)]

        return {
            "info": {"description": "TACO YOLO Dataset", "version": "1.0", "year": 2024},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch]
    }


def evaluate_coco_metrics(model, dataset, yolo_coco, processor, device, verbose=True):
    """Evaluate model and return COCO metrics (mAP, recall, etc.)"""
    model.eval()
    coco_gt = yolo_coco.coco
    results = []

    if verbose:
        print("Running COCO evaluation...")

    for idx in range(len(dataset)):
        img_path = dataset.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        predictions = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]

        for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
            x_min, y_min, x_max, y_max = box.cpu().tolist()
            results.append({
                "image_id": idx,
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

    def __init__(self, val_dataset, val_coco, processor, device):
        self.val_dataset = val_dataset
        self.val_coco = val_coco
        self.processor = processor
        self.device = device
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        print("\n" + "=" * 80)
        print(f"EPOCH {int(state.epoch)} METRICS")
        print("=" * 80)

        metrics = evaluate_coco_metrics(model, self.val_dataset, self.val_coco, self.processor, self.device, verbose=False)

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

# Configuration
IS_KAGGLE = os.path.exists("/kaggle")
DATASET_ROOT = "/kaggle/input/taco10-yolo" if IS_KAGGLE else "/home/mkultra/Documents/TACO/TACO/yolo_dataset"
OUTPUT_DIR = "/kaggle/working/rtdetr_output" if IS_KAGGLE else "./rtdetr_output"

CONFIG = {
    "model_name": "PekingU/rtdetr_v2_r50vd",
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-4,  # Increased from 1e-5 for better convergence
}

print("=" * 80)
print("RT-DETR v2 Training Pipeline (YOLO Format)")
print("=" * 80)
print(f"Dataset: {DATASET_ROOT}")
print(f"Model: {CONFIG['model_name']}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print()

# Load YOLO dataset configuration
dataset_root = Path(DATASET_ROOT)
with open(dataset_root / "data.yaml", 'r') as f:
    data_config = yaml.safe_load(f)

class_names = data_config['names']
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

print(f"Classes ({len(class_names)}): {class_names}")
print()

# Initialize
processor = RTDetrImageProcessor.from_pretrained(CONFIG['model_name'])
model = RTDetrV2ForObjectDetection.from_pretrained(
    CONFIG['model_name'],
    id2label=id2label,
    label2id=label2id,
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# Create datasets
train_dataset = YoloDetectionDataset(
    img_folder=dataset_root / "train" / "images",
    label_folder=dataset_root / "train" / "labels",
    processor=processor,
    class_names=class_names
)

val_dataset = YoloDetectionDataset(
    img_folder=dataset_root / "valid" / "images",
    label_folder=dataset_root / "valid" / "labels",
    processor=processor,
    class_names=class_names
)

# Create COCO format for evaluation
val_coco = YoloToCoco(
    img_folder=dataset_root / "valid" / "images",
    label_folder=dataset_root / "valid" / "labels",
    class_names=class_names
)

print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")
print()

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=CONFIG['epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    learning_rate=CONFIG['learning_rate'],
    warmup_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=50,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
)

# Setup COCO metrics callback
device = "cuda" if torch.cuda.is_available() else "cpu"
coco_callback = CocoEvalCallback(val_dataset, val_coco, processor, device)

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

metrics = evaluate_coco_metrics(model, val_dataset, val_coco, processor, device)

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
