#!/usr/bin/env python3
"""
TACO Dataset Preparation Script for RT-DETR v2 Training

This script prepares the TACO dataset by:
1. Creating train/validation/test splits
2. Converting 60 original categories to TACO-10 simplified categories
3. Generating COCO format annotations with 10 classes

Note: The original 60 classes don't provide good accuracy in practice.
We merge them into 10 categories for better training results.
"""

import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from collections import defaultdict


# TACO-10 categories (0-indexed)
TACO_10_CATEGORIES = [
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

# Mapping from original TACO 60 categories to TACO-10
CATEGORY_MAPPING = {
    "Aluminium foil": "Other",
    "Battery": "Other",
    "Aluminium blister pack": "Other",
    "Carded blister pack": "Other",
    "Other plastic bottle": "Bottle",
    "Clear plastic bottle": "Bottle",
    "Glass bottle": "Bottle",
    "Plastic bottle cap": "Bottle cap",
    "Metal bottle cap": "Bottle cap",
    "Broken glass": "Other",
    "Food Can": "Can",
    "Aerosol": "Can",
    "Drink can": "Can",
    "Toilet tube": "Other",
    "Other carton": "Other",
    "Egg carton": "Other",
    "Drink carton": "Other",
    "Corrugated carton": "Other",
    "Meal carton": "Other",
    "Pizza box": "Other",
    "Paper cup": "Cup",
    "Disposable plastic cup": "Cup",
    "Foam cup": "Cup",
    "Glass cup": "Cup",
    "Other plastic cup": "Cup",
    "Food waste": "Other",
    "Glass jar": "Other",
    "Plastic lid": "Lid",
    "Metal lid": "Lid",
    "Other plastic": "Other",
    "Magazine paper": "Other",
    "Tissues": "Other",
    "Wrapping paper": "Other",
    "Normal paper": "Other",
    "Paper bag": "Bag",
    "Plastified paper bag": "Bag",
    "Plastic film": "Other",
    "Six pack rings": "Other",
    "Garbage bag": "Bag",
    "Other plastic wrapper": "Other",
    "Single-use carrier bag": "Bag",
    "Polypropylene bag": "Bag",
    "Crisp packet": "Other",
    "Spread tub": "Other",
    "Tupperware": "Other",
    "Disposable food container": "Other",
    "Foam food container": "Other",
    "Other plastic container": "Other",
    "Plastic glooves": "Other",
    "Plastic utensils": "Other",
    "Pop tab": "Pop tab",
    "Rope & strings": "Other",
    "Scrap metal": "Other",
    "Shoe": "Other",
    "Squeezable tube": "Other",
    "Plastic straw": "Straw",
    "Paper straw": "Straw",
    "Styrofoam piece": "Other",
    "Unlabeled litter": "Other",
    "Cigarette": "Cigarette"
}


def get_taco10_class_id(original_category_name):
    """Map original TACO category name to TACO-10 class ID"""
    taco10_name = CATEGORY_MAPPING.get(original_category_name, "Other")
    return TACO_10_CATEGORIES.index(taco10_name)


def load_coco_annotations(annotation_file):
    """Load COCO format annotations"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data


def create_split_annotations(data, image_ids, split_name, original_id_to_name):
    """Create annotations for a specific split with TACO-10 categories"""
    # Filter images for this split
    split_images = []
    for img in data['images']:
        if img['id'] in image_ids:
            # Create a copy of the image info and update the file_name
            img_copy = img.copy()
            # Create unique filename using image ID and standardize to .jpg
            unique_filename = f"{img['id']:06d}.jpg"
            img_copy['file_name'] = unique_filename
            split_images.append(img_copy)

    # Convert annotations to TACO-10 categories
    split_annotations = []
    category_counts = defaultdict(int)

    for ann in data['annotations']:
        if ann['image_id'] in image_ids:
            # Get original category name
            original_cat_id = ann['category_id']
            original_cat_name = original_id_to_name.get(original_cat_id, "Unlabeled litter")

            # Map to TACO-10 class ID
            new_cat_id = get_taco10_class_id(original_cat_name)
            category_counts[new_cat_id] += 1

            # Create new annotation with updated category_id
            new_ann = ann.copy()
            new_ann['category_id'] = new_cat_id
            split_annotations.append(new_ann)

    # Create TACO-10 categories
    taco10_categories = [
        {"id": i, "name": name, "supercategory": name}
        for i, name in enumerate(TACO_10_CATEGORIES)
    ]

    # Create new annotation structure with TACO-10 categories
    split_data = {
        'info': {
            'description': f'TACO TACO-10 Dataset - {split_name} split',
            'version': '1.0',
            'year': 2024,
            'contributor': 'TACO Dataset',
            'date_created': data['info'].get('date_created', '')
        },
        'licenses': data.get('licenses', []),
        'categories': taco10_categories,
        'images': split_images,
        'annotations': split_annotations
    }

    return split_data, category_counts


def copy_images_to_split(data_dir, split_images, split_dir, original_images):
    """Copy images from batch directories to split directory"""
    split_dir.mkdir(parents=True, exist_ok=True)

    # Create a mapping from image ID to original file path
    id_to_original_path = {img['id']: img['file_name'] for img in original_images}

    for img in split_images:
        # Use the original file path (with batch prefix) to find the source
        original_file_path = id_to_original_path.get(img['id'])
        if original_file_path:
            src_path = data_dir / original_file_path
            dst_path = split_dir / img['file_name']  # Use the cleaned file name for destination

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
        else:
            print(f"Warning: Original path not found for image ID {img['id']}")


def prepare_taco_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Prepare TACO dataset for RT-DETR v2 training with TACO-10 categories

    Args:
        data_dir: Path to TACO data directory
        output_dir: Path to output directory for prepared dataset
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Load annotations
    print("=" * 80)
    print("TACO Dataset Preparation - TACO-10 (10 Classes)")
    print("=" * 80)
    print("Loading COCO annotations...")
    annotations_file = data_dir / 'annotations.json'
    data = load_coco_annotations(annotations_file)

    print(f"\nOriginal dataset:")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Original categories: {len(data['categories'])}")

    print(f"\nConverting to TACO-10 (10 simplified categories)...")
    print(f"Note: The original 60 classes don't provide good accuracy.")
    print(f"We merge them into 10 categories for better results.\n")

    # Create mapping from original category_id to category name
    original_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Get all image IDs
    image_ids = [img['id'] for img in data['images']]

    # Create train/val/test splits
    print("Creating train/validation/test splits...")
    train_ids, temp_ids = train_test_split(
        image_ids,
        test_size=(val_ratio + test_ratio),
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=42
    )

    print(f"Train: {len(train_ids)} images")
    print(f"Validation: {len(val_ids)} images")
    print(f"Test: {len(test_ids)} images")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'valid'
    test_dir = output_dir / 'test'

    # Process each split
    splits = {
        'train': (train_ids, train_dir),
        'valid': (val_ids, val_dir),
        'test': (test_ids, test_dir)
    }

    total_category_counts = defaultdict(int)

    for split_name, (split_ids, split_dir) in splits.items():
        print(f"\n{'=' * 80}")
        print(f"Processing {split_name} split...")
        print(f"{'=' * 80}")

        # Create annotations for this split with TACO-10 mapping
        split_data, category_counts = create_split_annotations(
            data, set(split_ids), split_name, original_id_to_name
        )

        # Accumulate category counts
        for cat_id, count in category_counts.items():
            total_category_counts[cat_id] += count

        # Save annotations
        split_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = split_dir / '_annotations.coco.json'
        with open(annotations_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        # Copy images
        copy_images_to_split(data_dir, split_data['images'], split_dir, data['images'])

        print(f"\n✓ Created {split_name} split:")
        print(f"  Images: {len(split_data['images'])}")
        print(f"  Annotations: {len(split_data['annotations'])}")
        print(f"  Saved to: {split_dir}")

        # Show category distribution for this split
        print(f"\n  Category distribution:")
        for cat_id in range(len(TACO_10_CATEGORIES)):
            count = category_counts[cat_id]
            if count > 0:
                print(f"    {TACO_10_CATEGORIES[cat_id]:20s}: {count:4d} annotations")

    # Create summary
    summary = {
        'dataset': 'TACO-10',
        'format': 'COCO',
        'num_classes': len(TACO_10_CATEGORIES),
        'total_images': len(image_ids),
        'total_annotations': sum(len(data['annotations']) for data in [
            load_coco_annotations(train_dir / '_annotations.coco.json'),
            load_coco_annotations(val_dir / '_annotations.coco.json'),
            load_coco_annotations(test_dir / '_annotations.coco.json')
        ]),
        'splits': {
            'train': len(train_ids),
            'valid': len(val_ids),
            'test': len(test_ids)
        },
        'categories': TACO_10_CATEGORIES
    }

    summary_path = output_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Dataset preparation completed!")
    print(f"{'=' * 80}")
    print(f"\nTACO-10 Categories ({len(TACO_10_CATEGORIES)}):")
    for i, cat in enumerate(TACO_10_CATEGORIES):
        count = total_category_counts[i]
        print(f"  {i}: {cat:20s} - {count:4d} annotations")

    print(f"\nOutput directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nYou can now train with:")
    print(f"  python train_rtdetrv2_coco.py")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Prepare TACO dataset with TACO-10 categories for RT-DETR v2 training'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/mkultra/Documents/TACO/data',
        help='Path to TACO data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/mkultra/Documents/TACO/dataset',
        help='Path to output directory'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Ratio of data for training'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Ratio of data for validation'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Ratio of data for testing'
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    prepare_taco_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()
