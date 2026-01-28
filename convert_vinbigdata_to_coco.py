"""
Convert VinBigData chest X-ray dataset to COCO format for Co-DETR training.
This script converts the VinBigData CSV annotations to COCO JSON format.
Handles DICOM medical images - they remain as .dicom, no conversion needed.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Try to import pydicom for DICOM support
try:
    import pydicom
    import numpy as np
    DICOM_SUPPORT = True
except ImportError:
    DICOM_SUPPORT = False
    print("Warning: pydicom not installed. Install with: pip install pydicom")


# Class mapping for VinBigData dataset
CLASS_MAPPING = {
    'Aortic enlargement': 0,
    'Atelectasis': 1,
    'Calcification': 2,
    'Cardiomegaly': 3,
    'Consolidation': 4,
    'ILD': 5,
    'Infiltration': 6,
    'Lung Opacity': 7,
    'Nodule/Mass': 8,
    'Other lesion': 9,
    'Pleural effusion': 10,
    'Pleural thickening': 11,
    'Pneumothorax': 12,
    'Pulmonary fibrosis': 13,
    # 'No finding': 14  # Exclude "No finding" class for object detection
}


def get_image_info(img_path, img_id):
    """Get image dimensions and create image info dict."""
    file_name = os.path.basename(img_path)

    # Check if it's a DICOM file
    if img_path.endswith('.dicom'):
        if not DICOM_SUPPORT:
            raise ImportError("pydicom is required to process DICOM files. Install with: pip install pydicom")

        dcm = pydicom.dcmread(img_path)
        img_array = dcm.pixel_array
        height, width = img_array.shape
    else:
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size

    return {
        'id': img_id,
        'file_name': file_name,
        'width': width,
        'height': height
    }


def convert_to_coco(csv_path, img_dir, output_path, split='train'):
    """
    Convert VinBigData CSV annotations to COCO format.

    Args:
        csv_path: Path to train.csv or test annotations
        img_dir: Directory containing images
        output_path: Output path for COCO JSON file
        split: 'train' or 'val'
    """
    print(f"Converting {split} set to COCO format...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter out "No finding" entries (they have NaN bounding boxes)
    df = df[df['class_name'] != 'No finding'].copy()
    df = df.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])

    # Initialize COCO format structure
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Add categories
    for class_name, class_id in CLASS_MAPPING.items():
        coco_format['categories'].append({
            'id': class_id,
            'name': class_name,
            'supercategory': 'abnormality'
        })

    # Get unique images
    unique_images = df['image_id'].unique()
    print(f"Found {len(unique_images)} unique images with annotations")

    # Create image_id mapping
    image_id_map = {img_name: idx for idx, img_name in enumerate(unique_images)}

    annotation_id = 0

    # Process each image
    for img_name in tqdm(unique_images, desc=f"Processing {split} images"):
        # Try different extensions
        img_path = None
        for ext in ['.dicom', '.png', '.jpg', '.jpeg', '.dcm']:
            test_path = os.path.join(img_dir, f"{img_name}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None:
            print(f"Warning: Image {img_name} not found, skipping...")
            continue

        # Get image info
        img_id = image_id_map[img_name]
        try:
            img_info = get_image_info(img_path, img_id)
            coco_format['images'].append(img_info)
        except Exception as e:
            print(f"Warning: Could not process image {img_name}: {e}")
            continue

        # Get all annotations for this image
        img_annotations = df[df['image_id'] == img_name]

        for _, row in img_annotations.iterrows():
            class_name = row['class_name']

            # Skip if class not in mapping
            if class_name not in CLASS_MAPPING:
                continue

            # Get bounding box coordinates
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])

            # Convert to COCO format [x, y, width, height]
            width = x_max - x_min
            height = y_max - y_min

            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue

            annotation = {
                'id': annotation_id,
                'image_id': img_id,
                'category_id': CLASS_MAPPING[class_name],
                'bbox': [x_min, y_min, width, height],
                'area': width * height,
                'iscrowd': 0
            }

            coco_format['annotations'].append(annotation)
            annotation_id += 1

    # Save COCO format JSON
    print(f"Saving COCO format to {output_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Categories: {len(coco_format['categories'])}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_format, f)

    print(f"Conversion complete!")
    return coco_format


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VinBigData to COCO format')
    parser.add_argument('--data_root', type=str,
                        default='/scratch/dpraja12/data/VinBigData',
                        help='Root directory of VinBigData dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations',
                        help='Output directory for COCO annotations')
    parser.add_argument('--train_ratio', type=float, default=0.85,
                        help='Ratio of training data (rest will be validation)')

    args = parser.parse_args()

    # Check if pydicom is installed
    if not DICOM_SUPPORT:
        print("\nERROR: pydicom is not installed!")
        print("The VinBigData dataset uses DICOM format.")
        print("\nPlease install pydicom:")
        print("  pip install pydicom")
        print("\nThen run this script again.")
        exit(1)

    # Paths
    train_csv = os.path.join(args.data_root, 'train.csv')
    train_img_dir = os.path.join(args.data_root, 'train')

    print("="*60)
    print("VinBigData to COCO Conversion")
    print("="*60)

    # Read and split the training data
    print(f"\nSplitting training data with ratio {args.train_ratio}")
    df = pd.read_csv(train_csv)

    # Get unique images (excluding "No finding" images without boxes)
    df_with_boxes = df[df['class_name'] != 'No finding'].copy()
    unique_images = df_with_boxes['image_id'].unique()

    # Split images
    import numpy as np
    np.random.seed(42)
    shuffled_images = np.random.permutation(unique_images)

    split_idx = int(len(shuffled_images) * args.train_ratio)
    train_images = set(shuffled_images[:split_idx])
    val_images = set(shuffled_images[split_idx:])

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    # Create temporary CSV files for train/val splits
    train_df = df[df['image_id'].isin(train_images)]
    val_df = df[df['image_id'].isin(val_images)]

    # Convert train set
    train_output = os.path.join(args.output_dir, 'instances_train.json')
    convert_to_coco(
        csv_path=train_csv,
        img_dir=train_img_dir,
        output_path=train_output,
        split='train'
    )

    # For validation, we need to filter the DataFrame
    val_csv_temp = '/tmp/val_temp.csv'
    val_df.to_csv(val_csv_temp, index=False)

    val_output = os.path.join(args.output_dir, 'instances_val.json')
    convert_to_coco(
        csv_path=val_csv_temp,
        img_dir=train_img_dir,
        output_path=val_output,
        split='val'
    )

    # Clean up temporary file
    if os.path.exists(val_csv_temp):
        os.remove(val_csv_temp)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nAnnotation files created:")
    print(f"  - Train: {train_output}")
    print(f"  - Val: {val_output}")
    print(f"\nNote: 14 abnormality classes (excluding 'No finding')")
    print(f"Images are in DICOM format and will be loaded during training.")
