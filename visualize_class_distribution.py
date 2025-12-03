#!/usr/bin/env python3
"""
YOLO Dataset Class Distribution Visualizer
Analyzes and visualizes the distribution of classes across train/val/test splits
"""

import yaml
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_yaml_config(yaml_path):
    """Load YOLO dataset configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_classes_in_labels(label_dir):
    """Count class instances in all label files"""
    class_counts = defaultdict(int)
    label_files = list(Path(label_dir).glob('*.txt'))

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    return class_counts, len(label_files)


def analyze_dataset(dataset_root):
    """Analyze class distribution across all splits"""
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / 'data.yaml'

    # Load configuration
    config = load_yaml_config(yaml_path)
    class_names = config['names']
    num_classes = config['nc']

    print(f"Dataset: {dataset_root.name}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}\n")

    # Analyze each split
    splits = ['train', 'valid', 'test']
    split_data = {}

    for split in splits:
        label_dir = dataset_root / split / 'labels'
        if label_dir.exists():
            class_counts, num_images = count_classes_in_labels(label_dir)
            split_data[split] = {
                'class_counts': class_counts,
                'num_images': num_images
            }

            print(f"{split.upper()} Split:")
            print(f"  Images: {num_images}")
            print(f"  Total instances: {sum(class_counts.values())}")
            print(f"  Classes present: {len(class_counts)}/{num_classes}\n")

    return class_names, split_data


def visualize_distribution(class_names, split_data, output_dir=None):
    """Create comprehensive visualizations of class distribution"""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('YOLO Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Combined distribution across all splits (bar chart)
    ax1 = axes[0, 0]
    all_counts = defaultdict(int)
    for split_info in split_data.values():
        for class_id, count in split_info['class_counts'].items():
            all_counts[class_id] += count

    class_ids = sorted(all_counts.keys())
    counts = [all_counts[i] for i in class_ids]
    labels = [class_names[i] for i in class_ids]

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_ids)))
    bars = ax1.bar(labels, counts, color=colors)
    ax1.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Instances', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # 2. Distribution by split (grouped bar chart)
    ax2 = axes[0, 1]
    splits = list(split_data.keys())
    x = np.arange(len(class_names))
    width = 0.25

    for idx, split in enumerate(splits):
        split_counts = [split_data[split]['class_counts'].get(i, 0) for i in range(len(class_names))]
        offset = width * (idx - len(splits)/2 + 0.5)
        ax2.bar(x + offset, split_counts, width, label=split.capitalize())

    ax2.set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Instances', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Pie chart of overall distribution
    ax3 = axes[1, 0]
    if counts:
        wedges, texts, autotexts = ax3.pie(counts, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax3.set_title('Class Proportion (All Splits)', fontsize=14, fontweight='bold')

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Class', 'Train', 'Valid', 'Test', 'Total', '%'])

    for i, class_name in enumerate(class_names):
        row = [class_name]
        total = 0
        for split in ['train', 'valid', 'test']:
            if split in split_data:
                count = split_data[split]['class_counts'].get(i, 0)
                row.append(str(count))
                total += count
            else:
                row.append('0')

        row.append(str(total))
        percentage = (total / sum(all_counts.values()) * 100) if sum(all_counts.values()) > 0 else 0
        row.append(f'{percentage:.1f}%')
        table_data.append(row)

    # Add totals row
    totals_row = ['TOTAL']
    for split in ['train', 'valid', 'test']:
        if split in split_data:
            total = sum(split_data[split]['class_counts'].values())
            totals_row.append(str(total))
        else:
            totals_row.append('0')
    totals_row.append(str(sum(all_counts.values())))
    totals_row.append('100%')
    table_data.append(totals_row)

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style totals row
    for i in range(len(table_data[0])):
        table[(len(table_data)-1, i)].set_facecolor('#d3d3d3')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')

    ax4.set_title('Detailed Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / 'class_distribution.png'
    else:
        output_file = 'class_distribution.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    plt.show()


def print_detailed_stats(class_names, split_data):
    """Print detailed statistics"""
    print("\n" + "="*70)
    print("DETAILED CLASS STATISTICS")
    print("="*70)

    all_counts = defaultdict(int)
    for split_info in split_data.values():
        for class_id, count in split_info['class_counts'].items():
            all_counts[class_id] += count

    total_instances = sum(all_counts.values())

    for i, class_name in enumerate(class_names):
        count = all_counts.get(i, 0)
        percentage = (count / total_instances * 100) if total_instances > 0 else 0

        print(f"\n{class_name} (Class {i}):")
        print(f"  Total instances: {count} ({percentage:.2f}%)")

        for split, split_info in split_data.items():
            split_count = split_info['class_counts'].get(i, 0)
            split_total = sum(split_info['class_counts'].values())
            split_pct = (split_count / split_total * 100) if split_total > 0 else 0
            print(f"  {split.capitalize()}: {split_count} ({split_pct:.2f}%)")

    print("\n" + "="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize YOLO dataset class distribution')
    parser.add_argument('dataset_path', type=str,
                       help='Path to YOLO dataset directory containing data.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualization (default: current directory)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot, only save it')

    args = parser.parse_args()

    # Analyze dataset
    class_names, split_data = analyze_dataset(args.dataset_path)

    # Print detailed statistics
    print_detailed_stats(class_names, split_data)

    # Create visualizations
    if args.no_show:
        plt.ioff()

    visualize_distribution(class_names, split_data, args.output)
