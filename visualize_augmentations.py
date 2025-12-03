"""
Visualize All RT-DETR Augmentations
Shows what each augmentation preset does to the same image
NO TRAINING - JUST VISUALIZATION!
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only for visualization

import argparse
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving images
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import albumentations as A
from pycocotools.coco import COCO


# TACO-10 colors (distinct colors for each class)
CLASS_COLORS = [
    '#FF6B6B',  # Bottle - Red
    '#4ECDC4',  # Can - Teal
    '#45B7D1',  # Cup - Blue
    '#96CEB4',  # Food wrapper - Green
    '#FFEAA7',  # Lid - Yellow
    '#DFE6E9',  # Other - Gray
    '#A29BFE',  # Paper - Purple
    '#FD79A8',  # Plastic bag - Pink
    '#FDCB6E',  # Straw - Orange
    '#6C5CE7',  # Cigarette - Indigo
]

TACO_10_CLASSES = [
    "Bottle",
    "Bottle cap",
    "Can",
    "Cigarette",
    "Cup",
    "Lid",
    "Other",
    "Bag",
    "Pop tab",
    "Straw"
]


def clip_boxes_to_image(boxes, img_width, img_height, min_size=1.0):
    """Clip COCO boxes to image bounds and filter out tiny boxes."""
    clipped_boxes = []
    valid_indices = []

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        # Clip to image bounds
        x = max(0.0, min(float(x), img_width))
        y = max(0.0, min(float(y), img_height))
        w = max(0.0, min(float(w), img_width - x))
        h = max(0.0, min(float(h), img_height - y))

        # Keep only boxes that are large enough
        if w >= min_size and h >= min_size:
            clipped_boxes.append([x, y, w, h])
            valid_indices.append(idx)

    return clipped_boxes, valid_indices


def cutmix_augmentation(image1, boxes1, labels1, image2, boxes2, labels2, p=1.0, min_bbox_area_ratio=0.1):
    """Apply YOLO-style CutMix augmentation"""
    if random.random() > p:
        return image1, boxes1, labels1

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    if (h1, w1) != (h2, w2):
        scale_x, scale_y = w1 / w2, h1 / h2
        image2 = cv2.resize(image2, (w1, h1))
        boxes2 = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in boxes2]

    cut_ratio = random.uniform(0.2, 0.5)
    cut_w = int(w1 * cut_ratio)
    cut_h = int(h1 * cut_ratio)

    cx = random.randint(0, w1 - cut_w)
    cy = random.randint(0, h1 - cut_h)

    x1, y1 = cx, cy
    x2, y2 = cx + cut_w, cy + cut_h

    for box in boxes1:
        bx, by, bw, bh = box
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(bx, x1)
        inter_y1 = max(by, y1)
        inter_x2 = min(bx2, x2)
        inter_y2 = min(by2, y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return image1, boxes1, labels1

    mixed_img = image1.copy()
    mixed_img[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    mixed_boxes = list(boxes1)
    mixed_labels = list(labels1)

    for box, label in zip(boxes2, labels2):
        bx, by, bw, bh = box
        bx2, by2 = bx + bw, by + bh
        original_area = bw * bh

        inter_x1 = max(bx, x1)
        inter_y1 = max(by, y1)
        inter_x2 = min(bx2, x2)
        inter_y2 = min(by2, y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            inter_area = inter_w * inter_h

            if inter_area / original_area >= min_bbox_area_ratio:
                new_x = inter_x1
                new_y = inter_y1
                new_w = inter_w
                new_h = inter_h

                if new_w > 2 and new_h > 2:
                    mixed_boxes.append([new_x, new_y, new_w, new_h])
                    mixed_labels.append(label)

    return mixed_img, mixed_boxes, mixed_labels


def get_augmentation_preset(preset_name="none", img_size=640):
    """Get augmentation pipeline for different strategies"""
    aug_transforms = []

    if preset_name == "flip":
        aug_transforms = [A.HorizontalFlip(p=1.0)]

    elif preset_name == "rotation":
        aug_transforms = [A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=1.0)]

    elif preset_name == "shear":
        aug_transforms = [A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=1.0)]

    elif preset_name == "hsv":
        aug_transforms = [
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7 * 255),
                val_shift_limit=int(0.4 * 255),
                p=0.5
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=1.0),
        ]

    elif preset_name == "blur":
        # Blur augmentation (matching YOLO config, p=1.0 for visualization)
        aug_transforms = [
            A.Blur(p=1.0, blur_limit=(3, 7)),
            A.MedianBlur(p=1.0, blur_limit=(3, 7)),
            A.ToGray(p=1.0),
            A.CLAHE(p=1.0, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
        ]

    elif preset_name == "mosaic":
        aug_transforms = [
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(img_size, img_size),
                cell_shape=(int(img_size * 0.6), int(img_size * 0.6)),
                center_range=(0.3, 0.7),
                fit_mode="cover",
                p=1.0
            ),
        ]

    elif preset_name == "all":
        aug_transforms = [
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=1.0),
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7 * 255),
                val_shift_limit=int(0.4 * 255),
                p=1.0
            ),
            A.Blur(p=1.0, blur_limit=(3, 7)),
            A.MedianBlur(p=1.0, blur_limit=(3, 7)),
            A.ToGray(p=1.0),
            A.CLAHE(p=1.0, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(img_size, img_size),
                cell_shape=(int(img_size * 0.6), int(img_size * 0.6)),
                center_range=(0.3, 0.7),
                fit_mode="cover",
                p=1.0
            ),
        ]

    elif preset_name == "cutmix":
        # CutMix is handled separately, not an Albumentations transform
        aug_transforms = []

    elif preset_name == "none":
        aug_transforms = []

    else:
        raise ValueError(f"Unknown augmentation preset: {preset_name}")

    if len(aug_transforms) == 0:
        return None

    # Use COCO format (x, y, width, height) - no conversion needed!
    transform = A.Compose(
        aug_transforms,
        bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,
        )
    )

    return transform


def apply_augmentation(image_np, boxes, labels, aug_name, aug_transform, coco, image_ids, img_id, train_dir):
    """Apply a specific augmentation to an image"""
    aug_image = image_np.copy()
    aug_boxes = boxes.copy()
    aug_labels = labels.copy()
    aug_h, aug_w = aug_image.shape[:2]

    # Clip boxes to image bounds BEFORE augmentation
    aug_boxes, valid_indices = clip_boxes_to_image(aug_boxes, aug_w, aug_h)
    aug_labels = [aug_labels[i] for i in valid_indices]

    has_cutmix = aug_name in ['cutmix', 'all']

    # Prepare mosaic metadata if needed
    mosaic_metadata = []
    if aug_transform is not None and hasattr(aug_transform, 'transforms'):
        for t in aug_transform.transforms:
            if t.__class__.__name__ == 'Mosaic':
                additional_ids = random.sample([i for i in image_ids if i != img_id], min(3, len(image_ids) - 1))

                for add_id in additional_ids:
                    add_img_info = coco.imgs[add_id]
                    add_img_path = train_dir / add_img_info['file_name']
                    add_image = Image.open(add_img_path).convert("RGB")
                    add_image_np = np.array(add_image)

                    add_ann_ids = coco.getAnnIds(imgIds=add_id)
                    add_anns = coco.loadAnns(add_ann_ids)

                    # Collect bounding boxes in COCO format
                    add_boxes = []
                    add_labels = []
                    for ann in add_anns:
                        add_boxes.append(ann['bbox'])
                        add_labels.append(ann['category_id'])

                    # Clip mosaic boxes too
                    add_h, add_w = add_image_np.shape[:2]
                    add_boxes, add_valid = clip_boxes_to_image(add_boxes, add_w, add_h)
                    add_labels = [add_labels[i] for i in add_valid]

                    mosaic_metadata.append({
                        'image': add_image_np,
                        'bboxes': add_boxes,  # Keep in COCO format
                        'class_labels': add_labels
                    })
                break

    # Apply Albumentations transforms (now using COCO format directly)
    if aug_transform is not None:
        transform_data = {
            'image': aug_image,
            'bboxes': aug_boxes if len(aug_boxes) > 0 else [],  # COCO format
            'class_labels': aug_labels if len(aug_labels) > 0 else []
        }

        if mosaic_metadata:
            transform_data['mosaic_metadata'] = mosaic_metadata

        try:
            transformed = aug_transform(**transform_data)
            aug_image = transformed['image']
            aug_boxes = list(transformed.get('bboxes', []))  # Still in COCO format
            aug_labels = list(transformed.get('class_labels', []))
        except Exception as e:
            print(f"    Warning: {aug_name} augmentation failed: {e}")

    # Apply CutMix if needed
    if has_cutmix and len(image_ids) > 1:
        cutmix_id = random.choice([i for i in image_ids if i != img_id])
        cutmix_img_info = coco.imgs[cutmix_id]
        cutmix_img_path = train_dir / cutmix_img_info['file_name']
        cutmix_image = Image.open(cutmix_img_path).convert("RGB")
        cutmix_image_np = np.array(cutmix_image)

        cutmix_ann_ids = coco.getAnnIds(imgIds=cutmix_id)
        cutmix_anns = coco.loadAnns(cutmix_ann_ids)

        # Collect bounding boxes in COCO format
        cutmix_boxes = []
        cutmix_labels = []
        for ann in cutmix_anns:
            cutmix_boxes.append(ann['bbox'])
            cutmix_labels.append(ann['category_id'])

        aug_image, aug_boxes, aug_labels = cutmix_augmentation(
            aug_image, aug_boxes, aug_labels,
            cutmix_image_np, cutmix_boxes, cutmix_labels,
            p=1.0
        )

    # Clip boxes after all augmentations
    aug_h, aug_w = aug_image.shape[:2]
    aug_boxes, valid_indices = clip_boxes_to_image(aug_boxes, aug_w, aug_h)
    aug_labels = [aug_labels[i] for i in valid_indices]

    return aug_image, aug_boxes, aug_labels


def draw_boxes_on_axis(ax, image, boxes, labels, title):
    """Draw image with bounding boxes on a matplotlib axis"""
    ax.imshow(image)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    for box, label in zip(boxes, labels):
        x, y, w, h = box
        label = int(label)  # Ensure label is integer
        color = CLASS_COLORS[label % len(CLASS_COLORS)]

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        class_name = TACO_10_CLASSES[label] if label < len(TACO_10_CLASSES) else f"Class {label}"
        ax.text(
            x, y - 5,
            class_name,
            fontsize=8,
            color='white',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
        )


def visualize_all_augmentations(dataset_root, num_samples=3, output_dir="aug_visualizations"):
    """Visualize all augmentation presets on the same images"""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load COCO annotations
    train_dir = dataset_root / "train"
    annotation_file = train_dir / "_annotations.coco.json"

    if not annotation_file.exists():
        print(f"Error: Annotation file not found at {annotation_file}")
        return

    print("=" * 80)
    print("AUGMENTATION VISUALIZATION (NO TRAINING)")
    print("=" * 80)
    print("\nLoading annotations...")
    coco = COCO(str(annotation_file))
    image_ids = list(coco.imgs.keys())

    # Augmentation presets to visualize
    aug_presets = ['none', 'flip', 'rotation', 'shear', 'hsv', 'blur', 'mosaic', 'cutmix', 'all']

    print(f"\nAugmentations to visualize: {', '.join(aug_presets)}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}/\n")

    # Sample random images
    sampled_ids = random.sample(image_ids, min(num_samples, len(image_ids)))

    for sample_idx, img_id in enumerate(sampled_ids):
        print(f"[{sample_idx + 1}/{len(sampled_ids)}] Processing sample...")

        # Load image
        img_info = coco.imgs[img_id]
        img_path = train_dir / img_info['file_name']

        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Get image dimensions
        img_h, img_w = image_np.shape[:2]

        # Load and clip bounding boxes to image bounds
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Clip to image bounds
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            # Only keep valid boxes
            if w > 1 and h > 1:
                boxes.append([x, y, w, h])
                labels.append(ann['category_id'])

        # Create a grid: 3 rows x 3 cols (9 augmentations)
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        axes = axes.flatten()

        for idx, aug_name in enumerate(aug_presets):
            # Get augmentation transform
            aug_transform = get_augmentation_preset(aug_name, img_size=640)

            # Apply augmentation
            aug_image, aug_boxes, aug_labels = apply_augmentation(
                image_np, boxes, labels, aug_name, aug_transform, coco, image_ids, img_id, train_dir
            )

            # Draw on subplot
            title = f"{aug_name.upper()}\n({len(aug_boxes)} objects)"
            draw_boxes_on_axis(axes[idx], aug_image, aug_boxes, aug_labels, title)

        plt.suptitle(f"All Augmentation Presets - Sample {sample_idx + 1}",
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        output_path = output_dir / f"all_augmentations_sample_{sample_idx+1:02d}.png"
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    print(f"\n{'='*80}")
    print(f"Done! All visualizations saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize all RT-DETR augmentation presets")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Path to TACO dataset root")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of sample images to visualize (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./aug_visualizations",
                        help="Directory to save visualizations")

    args = parser.parse_args()

    visualize_all_augmentations(
        dataset_root=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
