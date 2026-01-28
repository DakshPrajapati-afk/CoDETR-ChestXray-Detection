#!/usr/bin/env python3
"""
Visualize detection predictions on images.
Shows best, worst, and sample predictions with ground truth comparison.

Usage:
    python visualize_predictions.py --results results.pkl --ann annotations.json --output vis_output/
"""

import argparse
import pickle
import os
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# VinBigData class names
CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

# Colors for each class (RGB)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (128, 255, 0)
]


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_dicom_image(filepath):
    """Load a DICOM image and convert to RGB numpy array."""
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required to load DICOM images")

    dcm = pydicom.dcmread(filepath)

    # Apply VOI LUT (windowing)
    img = apply_voi_lut(dcm.pixel_array, dcm)

    # Normalize to 0-255
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max() * 255
    img = img.astype(np.uint8)

    # Convert to RGB
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def load_image(filepath):
    """Load image from file (supports DICOM, PNG, JPG)."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.dcm', '.dicom', '']:
        # Try DICOM first
        try:
            return load_dicom_image(filepath)
        except:
            pass

    # Try standard image formats
    if HAS_CV2:
        img = cv2.imread(filepath)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if HAS_PIL:
        img = Image.open(filepath)
        return np.array(img.convert('RGB'))

    raise ValueError(f"Could not load image: {filepath}")


def draw_boxes_pil(img, gt_boxes, pred_boxes, gt_classes, pred_classes, pred_scores):
    """Draw bounding boxes on image using PIL. Returns side-by-side comparison."""

    # Create two copies of the image
    img_gt = Image.fromarray(img.copy())
    img_pred = Image.fromarray(img.copy())

    draw_gt = ImageDraw.Draw(img_gt)
    draw_pred = ImageDraw.Draw(img_pred)

    # Try to load a font - larger sizes for better visibility
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw ground truth boxes on left image
    for bbox, cls_id in zip(gt_boxes, gt_classes):
        x, y, w, h = bbox
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        # Draw rectangle with thicker lines
        draw_gt.rectangle([x, y, x + w, y + h], outline=color, width=4)
        # Draw label
        label = f"{CLASS_NAMES[cls_id]}" if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        # Draw text background
        text_y = max(0, y - 30)
        bbox_text = draw_gt.textbbox((x, text_y), label, font=font)
        draw_gt.rectangle([bbox_text[0]-2, bbox_text[1]-2, bbox_text[2]+2, bbox_text[3]+2], fill=(0, 0, 0))
        draw_gt.text((x, text_y), label, fill=color, font=font)

    # Draw prediction boxes on right image
    for bbox, cls_id, score in zip(pred_boxes, pred_classes, pred_scores):
        x1, y1, x2, y2 = bbox
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        # Draw rectangle with thicker lines
        draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=4)
        # Draw label with score
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}" if cls_id < len(CLASS_NAMES) else f"{cls_id}: {score:.2f}"
        # Draw text background
        text_y = max(0, y1 - 30)
        bbox_text = draw_pred.textbbox((x1, text_y), label, font=font)
        draw_pred.rectangle([bbox_text[0]-2, bbox_text[1]-2, bbox_text[2]+2, bbox_text[3]+2], fill=(0, 0, 0))
        draw_pred.text((x1, text_y), label, fill=color, font=font)

    # Add titles
    # Ground truth title
    gt_title = "GROUND TRUTH"
    bbox_title = draw_gt.textbbox((10, 10), gt_title, font=title_font)
    draw_gt.rectangle([bbox_title[0]-5, bbox_title[1]-5, bbox_title[2]+5, bbox_title[3]+5], fill=(0, 100, 0))
    draw_gt.text((10, 10), gt_title, fill=(255, 255, 255), font=title_font)

    # Prediction title
    pred_title = "PREDICTION"
    bbox_title = draw_pred.textbbox((10, 10), pred_title, font=title_font)
    draw_pred.rectangle([bbox_title[0]-5, bbox_title[1]-5, bbox_title[2]+5, bbox_title[3]+5], fill=(100, 0, 0))
    draw_pred.text((10, 10), pred_title, fill=(255, 255, 255), font=title_font)

    # Combine side by side
    width, height = img_gt.size
    combined = Image.new('RGB', (width * 2 + 10, height), color=(50, 50, 50))
    combined.paste(img_gt, (0, 0))
    combined.paste(img_pred, (width + 10, 0))

    return np.array(combined)


def calculate_image_metrics(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_threshold=0.5):
    """Calculate metrics for a single image."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'avg_iou': 0, 'score': 1.0}

    if len(gt_boxes) == 0:
        return {'tp': 0, 'fp': len(pred_boxes), 'fn': 0, 'avg_iou': 0, 'score': 0.0}

    if len(pred_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_boxes), 'avg_iou': 0, 'score': 0.0}

    gt_matched = [False] * len(gt_boxes)
    ious = []
    tp = 0

    # Sort predictions by score
    sorted_idx = np.argsort(pred_scores)[::-1]

    for idx in sorted_idx:
        pred_box = pred_boxes[idx]
        pred_cls = pred_classes[idx]
        pred_xywh = [pred_box[0], pred_box[1], pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            # Only match with GT of the same class
            if gt_classes[gt_idx] != pred_cls:
                continue
            iou = compute_iou(pred_xywh, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp += 1
            ious.append(best_iou)

    fp = len(pred_boxes) - tp
    fn = sum(1 for m in gt_matched if not m)
    avg_iou = np.mean(ious) if ious else 0

    # Calculate a combined score (F1-like with IoU bonus)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    score = f1 * 0.7 + avg_iou * 0.3  # Weighted combination

    return {'tp': tp, 'fp': fp, 'fn': fn, 'avg_iou': avg_iou, 'score': score,
            'precision': precision, 'recall': recall, 'f1': f1}


def visualize_predictions(results_file, ann_file, img_prefix, output_dir,
                          num_best=5, num_worst=5, num_random=5, score_threshold=0.3):
    """Generate visualization of predictions."""

    if not HAS_PIL:
        raise ImportError("PIL/Pillow is required for visualization")

    # Load data
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'best'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'worst'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'random'), exist_ok=True)

    print(f"Calculating metrics for {len(img_ids)} images...")

    # Calculate metrics for each image
    image_metrics = []

    for idx, img_id in enumerate(img_ids):
        # Get image info
        img_info = coco.loadImgs(img_id)[0]

        # Get ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in gt_anns]
        gt_classes = [ann['category_id'] for ann in gt_anns]  # Already 0-indexed in VinBigData

        # Get predictions above threshold
        pred_boxes = []
        pred_classes = []
        pred_scores = []

        for cls_id, class_preds in enumerate(results[idx]):
            for pred in class_preds:
                if pred[4] >= score_threshold:
                    pred_boxes.append(pred[:4])
                    pred_classes.append(cls_id)
                    pred_scores.append(pred[4])

        # Calculate metrics
        metrics = calculate_image_metrics(gt_boxes, gt_classes, pred_boxes,
                                          pred_classes, pred_scores)

        image_metrics.append({
            'idx': idx,
            'img_id': img_id,
            'img_info': img_info,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'pred_boxes': pred_boxes,
            'pred_classes': pred_classes,
            'pred_scores': pred_scores,
            'metrics': metrics
        })

    # Sort by score
    image_metrics.sort(key=lambda x: x['metrics']['score'], reverse=True)

    # Select images to visualize
    best_images = image_metrics[:num_best]
    worst_images = image_metrics[-num_worst:]

    # Random sample from middle
    middle_start = num_best
    middle_end = len(image_metrics) - num_worst
    if middle_end > middle_start:
        random_indices = np.random.choice(range(middle_start, middle_end),
                                          min(num_random, middle_end - middle_start),
                                          replace=False)
        random_images = [image_metrics[i] for i in random_indices]
    else:
        random_images = []

    def save_visualization(img_data, category, rank):
        """Save visualization for one image."""
        img_info = img_data['img_info']

        # Construct image path
        filename = img_info['file_name']
        img_path = os.path.join(img_prefix, filename)

        # Try different extensions if file not found
        if not os.path.exists(img_path):
            base = os.path.splitext(filename)[0]
            for ext in ['', '.dicom', '.dcm', '.png', '.jpg']:
                test_path = os.path.join(img_prefix, base + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            return

        # Load image
        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"  Warning: Could not load {img_path}: {e}")
            return

        # Draw boxes
        img_vis = draw_boxes_pil(img,
                                 img_data['gt_boxes'],
                                 img_data['pred_boxes'],
                                 img_data['gt_classes'],
                                 img_data['pred_classes'],
                                 img_data['pred_scores'])

        # Add metrics text
        metrics = img_data['metrics']
        img_pil = Image.fromarray(img_vis)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40 )
        except:
            font = ImageFont.load_default()

        # Add metrics overlay
        text_lines = [
            f"Score: {metrics['score']:.3f}",
            f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']}",
            f"Avg IoU: {metrics['avg_iou']:.3f}",
            f"Prec: {metrics.get('precision', 0):.3f} Rec: {metrics.get('recall', 0):.3f}"
        ]

        y_pos = 60
        for line in text_lines:
            # Draw text with background
            bbox = draw.textbbox((10, y_pos), line, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
            draw.text((10, y_pos), line, fill=(255, 255, 255), font=font)
            y_pos += 35

        # Save
        output_path = os.path.join(output_dir, category, f"{rank:02d}_{img_info['file_name'].replace('.dicom', '.png')}")
        img_pil.save(output_path)
        print(f"  Saved: {output_path}")

    # Generate visualizations
    print(f"\nGenerating {num_best} best predictions...")
    for rank, img_data in enumerate(best_images, 1):
        save_visualization(img_data, 'best', rank)

    print(f"\nGenerating {num_worst} worst predictions...")
    for rank, img_data in enumerate(worst_images, 1):
        save_visualization(img_data, 'worst', rank)

    print(f"\nGenerating {len(random_images)} random predictions...")
    for rank, img_data in enumerate(random_images, 1):
        save_visualization(img_data, 'random', rank)

    # Print summary
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nBest predictions (highest scores):")
    for i, img_data in enumerate(best_images, 1):
        m = img_data['metrics']
        print(f"  {i}. {img_data['img_info']['file_name']}: score={m['score']:.3f}, IoU={m['avg_iou']:.3f}")

    print(f"\nWorst predictions (lowest scores):")
    for i, img_data in enumerate(worst_images, 1):
        m = img_data['metrics']
        print(f"  {i}. {img_data['img_info']['file_name']}: score={m['score']:.3f}, IoU={m['avg_iou']:.3f}")

    print(f"\nLegend:")
    print(f"  GREEN boxes = Ground Truth")
    print(f"  COLORED boxes = Predictions (color by class)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize detection predictions')
    parser.add_argument('--results', type=str,
                        default='work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results.pkl',
                        help='Path to results.pkl')
    parser.add_argument('--ann', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations/instances_val.json',
                        help='Path to annotations')
    parser.add_argument('--img-prefix', type=str,
                        default='/scratch/dpraja12/data/VinBigData/train/',
                        help='Path to images directory')
    parser.add_argument('--output', type=str,
                        default='visualization_output',
                        help='Output directory for visualizations')
    parser.add_argument('--num-best', type=int, default=10,
                        help='Number of best predictions to visualize')
    parser.add_argument('--num-worst', type=int, default=10,
                        help='Number of worst predictions to visualize')
    parser.add_argument('--num-random', type=int, default=10,
                        help='Number of random predictions to visualize')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Score threshold for predictions')

    args = parser.parse_args()

    visualize_predictions(
        args.results, args.ann, args.img_prefix, args.output,
        args.num_best, args.num_worst, args.num_random, args.score_threshold
    )


if __name__ == '__main__':
    main()
