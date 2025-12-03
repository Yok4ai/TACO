#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format with TACO-10 category mapping
"""
import json
import os
from pathlib import Path
from tqdm import tqdm


# TACO-10 categories (0-indexed for YOLO)
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
# Index is the original category name, value is the TACO-10 category
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
    """
    Map original TACO category name to TACO-10 class ID
    """
    taco10_name = CATEGORY_MAPPING.get(original_category_name, "Other")
    return TACO_10_CATEGORIES.index(taco10_name)


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All values normalized to [0, 1]
    """
    x, y, w, h = coco_bbox

    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    width = w / img_width
    height = h / img_height

    return x_center, y_center, width, height


def convert_split(split_dir, output_dir, original_categories):
    """
    Convert a single split (train/valid/test) from COCO to YOLO format
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    # Create output directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    coco_file = split_dir / '_annotations.coco.json'
    print(f"Loading {coco_file}...")

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create mapping from COCO category_id to category name
    coco_id_to_name = {}
    for cat in coco_data['categories']:
        coco_id_to_name[cat['id']] = cat['name']

    # Create image id to filename and dimensions mapping
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Convert annotations
    print(f"Converting {len(image_info)} images...")
    for img_id, info in tqdm(image_info.items()):
        img_file = info['file_name']
        img_width = info['width']
        img_height = info['height']

        # Copy image
        src_img = split_dir / img_file
        dst_img = images_dir / img_file
        if src_img.exists():
            if not dst_img.exists():
                import shutil
                shutil.copy2(src_img, dst_img)

        # Create YOLO label file
        label_file = labels_dir / (Path(img_file).stem + '.txt')

        with open(label_file, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    # Get original category name
                    coco_cat_id = ann['category_id']
                    original_cat_name = coco_id_to_name[coco_cat_id]

                    # Map to TACO-10 class ID
                    class_id = get_taco10_class_id(original_cat_name)

                    # Convert bbox
                    coco_bbox = ann['bbox']
                    yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)

                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

    print(f"Completed {split_dir.name} split")
    return len(image_info)


def create_yaml_config(dataset_dir):
    """
    Create data.yaml configuration file for YOLO
    """
    yaml_content = f"""# TACO-10 Dataset YOLO Configuration
path: {dataset_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes (TACO-10)
nc: {len(TACO_10_CATEGORIES)}  # number of classes
names: {TACO_10_CATEGORIES}  # class names
"""

    yaml_file = Path(dataset_dir) / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_file}")


def main():
    # Paths
    base_dataset_dir = Path('/home/mkultra/Documents/TACO/TACO/dataset')
    output_base_dir = Path('/home/mkultra/Documents/TACO/TACO/yolo_dataset')

    # Load dataset summary for original category names
    summary_file = base_dataset_dir / 'dataset_summary.json'
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    original_categories = summary['category_names']

    print(f"Converting TACO dataset from {base_dataset_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Original categories: {len(original_categories)}")
    print(f"TACO-10 categories: {len(TACO_10_CATEGORIES)}")
    print(f"Categories: {TACO_10_CATEGORIES}\n")

    # Convert each split
    for split in ['train', 'valid', 'test']:
        split_dir = base_dataset_dir / split
        output_dir = output_base_dir / split

        if split_dir.exists():
            convert_split(split_dir, output_dir, original_categories)
        else:
            print(f"Warning: {split_dir} does not exist, skipping...")

    # Create YAML config
    create_yaml_config(output_base_dir)

    print("\n✓ Conversion complete!")
    print(f"YOLO dataset created at: {output_base_dir}")
    print(f"\nTACO-10 Categories ({len(TACO_10_CATEGORIES)}):")
    for i, cat in enumerate(TACO_10_CATEGORIES):
        print(f"  {i}: {cat}")


if __name__ == '__main__':
    main()
