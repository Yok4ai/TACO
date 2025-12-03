#!/usr/bin/env python3
"""
TACO Dataset Preparation Script for RF-DETR Training

This script prepares the TACO dataset for training with RF-DETR by:
1. Creating train/validation/test splits
2. Converting to the expected directory structure
3. Splitting the main annotations.json into separate files
"""

import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def load_coco_annotations(annotation_file):
    """Load COCO format annotations"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data


def create_split_annotations(data, image_ids, split_name):
    """Create annotations for a specific split"""
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

    # Filter annotations for this split
    split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]

    # Create new annotation structure
    split_data = {
        'info': data['info'],
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': split_images,
        'annotations': split_annotations
    }

    return split_data


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
    Prepare TACO dataset for RF-DETR training

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
    print("Loading COCO annotations...")
    annotations_file = data_dir / 'annotations.json'
    data = load_coco_annotations(annotations_file)

    print(f"Found {len(data['images'])} images and {len(data['annotations'])} annotations")
    print(f"Categories: {len(data['categories'])}")

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

    for split_name, (split_ids, split_dir) in splits.items():
        print(f"\nProcessing {split_name} split...")

        # Create annotations for this split
        split_data = create_split_annotations(data, set(split_ids), split_name)

        # Save annotations
        split_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = split_dir / '_annotations.coco.json'
        with open(annotations_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        # Copy images
        copy_images_to_split(data_dir, split_data['images'], split_dir, data['images'])

        print(f"Created {split_name} split with {len(split_data['images'])} images")
        print(f"  Annotations saved to: {annotations_path}")
        print(f"  Images copied to: {split_dir}")

    # Create summary
    summary = {
        'total_images': len(image_ids),
        'total_annotations': len(data['annotations']),
        'categories': len(data['categories']),
        'splits': {
            'train': len(train_ids),
            'valid': len(val_ids),
            'test': len(test_ids)
        },
        'category_names': [cat['name'] for cat in data['categories']]
    }

    summary_path = output_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Prepare TACO dataset for RF-DETR training')
    parser.add_argument('--data_dir', type=str, default='/home/mkultra/Documents/TACO/TACO/data',
                        help='Path to TACO data directory')
    parser.add_argument('--output_dir', type=str, default='/home/mkultra/Documents/TACO/TACO/dataset',
                        help='Path to output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of data for testing')

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