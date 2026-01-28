#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for object detection models.
Calculates: mAP, FROC, Average IoU, Precision, Recall, F1

Usage:
    python evaluate_model.py --results results.pkl --ann annotations.json
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# VinBigData class names
CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
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


def calculate_all_metrics(results_file, ann_file, iou_threshold=0.5, score_threshold=0.3):
    """
    Calculate comprehensive evaluation metrics.

    Returns dict with: mAP, FROC, avg_iou, precision, recall, f1
    """
    # Load data
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loading annotations from {ann_file}...")
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    num_images = len(img_ids)
    num_classes = len(results[0]) if results else 14

    print(f"Evaluating {num_images} images, {num_classes} classes...")

    # ========== Calculate mAP using COCO API ==========
    print("\n" + "="*60)
    print("COCO-STYLE mAP EVALUATION")
    print("="*60)

    # Convert results to COCO format
    coco_results = []
    for idx, img_id in enumerate(img_ids):
        for cls_id, class_preds in enumerate(results[idx]):
            for pred in class_preds:
                coco_results.append({
                    'image_id': img_id,
                    'category_id': cls_id,  # VinBigData uses 0-indexed
                    'bbox': [pred[0], pred[1], pred[2] - pred[0], pred[3] - pred[1]],  # xyxy to xywh
                    'score': float(pred[4])
                })

    if coco_results:
        # Ensure required fields exist in dataset
        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = {'description': 'VinBigData', 'version': '1.0'}
        if 'licenses' not in coco_gt.dataset:
            coco_gt.dataset['licenses'] = []

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_50_95 = coco_eval.stats[0]  # mAP@[0.5:0.95]
        map_50 = coco_eval.stats[1]     # mAP@0.5
        map_75 = coco_eval.stats[2]     # mAP@0.75
    else:
        map_50_95 = map_50 = map_75 = 0.0

    # ========== Calculate Average IoU, Precision, Recall, F1 ==========
    print("\n" + "="*60)
    print(f"DETECTION METRICS (IoU threshold={iou_threshold}, score threshold={score_threshold})")
    print("="*60)

    all_ious = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-class metrics
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []})

    for idx, img_id in enumerate(img_ids):
        # Get ground truth
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        # Group GT by class
        gt_by_class = defaultdict(list)
        for ann in gt_anns:
            cls_id = ann['category_id']  # VinBigData already 0-indexed
            gt_by_class[cls_id].append(ann['bbox'])

        # Process each class
        for cls_id in range(num_classes):
            gt_boxes = gt_by_class[cls_id]
            gt_matched = [False] * len(gt_boxes)

            # Get predictions for this class above score threshold
            preds = results[idx][cls_id]
            preds_filtered = [p for p in preds if p[4] >= score_threshold]

            # Sort by score
            preds_filtered.sort(key=lambda x: x[4], reverse=True)

            # Match predictions to GT
            for pred in preds_filtered:
                pred_xywh = [pred[0], pred[1], pred[2] - pred[0], pred[3] - pred[1]]

                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    iou = compute_iou(pred_xywh, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # True positive
                    gt_matched[best_gt_idx] = True
                    total_tp += 1
                    class_metrics[cls_id]['tp'] += 1
                    all_ious.append(best_iou)
                    class_metrics[cls_id]['ious'].append(best_iou)
                else:
                    # False positive
                    total_fp += 1
                    class_metrics[cls_id]['fp'] += 1

            # Count false negatives (unmatched GT)
            fn = sum(1 for m in gt_matched if not m)
            total_fn += fn
            class_metrics[cls_id]['fn'] += fn

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0

    print(f"\nOverall Metrics:")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  Average IoU:     {avg_iou:.4f}")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'Avg IoU':>8}")
    print("-" * 75)

    for cls_id in range(num_classes):
        m = class_metrics[cls_id]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        cls_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        cls_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        cls_iou = np.mean(m['ious']) if m['ious'] else 0
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        print(f"{class_name:<25} {tp:>6} {fp:>6} {fn:>6} {cls_prec:>8.4f} {cls_rec:>8.4f} {cls_iou:>8.4f}")

    # ========== Calculate FROC ==========
    print("\n" + "="*60)
    print("FROC EVALUATION (VinBigData Competition Metric)")
    print("="*60)

    # Collect all detections for FROC
    all_detections = []
    total_gt = 0

    for idx, img_id in enumerate(img_ids):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        # Group GT by class for class-aware matching
        gt_by_class = {}
        for ann in gt_anns:
            cls_id = ann['category_id']
            if cls_id not in gt_by_class:
                gt_by_class[cls_id] = {'boxes': [], 'matched': []}
            gt_by_class[cls_id]['boxes'].append(ann['bbox'])
            gt_by_class[cls_id]['matched'].append(False)

        total_gt += len(gt_anns)

        # Get all predictions
        all_preds = []
        for cls_id, class_preds in enumerate(results[idx]):
            for pred in class_preds:
                all_preds.append((pred[4], pred[:4], cls_id))

        all_preds.sort(key=lambda x: x[0], reverse=True)

        for score, pred_box, cls_id in all_preds:
            pred_xywh = [pred_box[0], pred_box[1], pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]]

            is_tp = False
            best_iou = iou_threshold
            best_gt_idx = -1

            # Only match with GT of the same class
            if cls_id in gt_by_class:
                gt_boxes = gt_by_class[cls_id]['boxes']
                gt_matched = gt_by_class[cls_id]['matched']

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    iou = compute_iou(pred_xywh, gt_box)
                    if iou >= best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    is_tp = True

            all_detections.append((score, is_tp))

    # Sort and calculate FROC curve
    all_detections.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    fps_per_image = []
    sensitivities = []

    for score, is_tp in all_detections:
        if is_tp:
            tp += 1
        else:
            fp += 1
        sensitivities.append(tp / total_gt if total_gt > 0 else 0)
        fps_per_image.append(fp / num_images)

    # FROC at standard FP rates
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_scores = []

    for fp_rate in fp_rates:
        sens = 0
        for i, fp_per_img in enumerate(fps_per_image):
            if fp_per_img <= fp_rate:
                sens = sensitivities[i]
        froc_scores.append(sens)

    avg_froc = np.mean(froc_scores)

    print(f"\nSensitivity at FP/image rates:")
    for fp_rate, sens in zip(fp_rates, froc_scores):
        print(f"  {fp_rate:6.3f} FP/img: {sens:.4f}")
    print(f"\nMean FROC Score: {avg_froc:.4f}")

    # ========== Summary ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  mAP@0.5:        {map_50:.4f}")
    print(f"  mAP@[.5:.95]:   {map_50_95:.4f}")
    print(f"  mAP@0.75:       {map_75:.4f}")
    print(f"  Mean FROC:      {avg_froc:.4f}")
    print(f"  Average IoU:    {avg_iou:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print("="*60 + "\n")

    return {
        'map_50': map_50,
        'map_50_95': map_50_95,
        'map_75': map_75,
        'avg_froc': avg_froc,
        'avg_iou': avg_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--results', type=str,
                        default='work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results.pkl',
                        help='Path to results.pkl')
    parser.add_argument('--ann', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations/instances_val.json',
                        help='Path to annotations')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--score', type=float, default=0.3,
                        help='Score threshold for predictions')

    args = parser.parse_args()

    metrics = calculate_all_metrics(args.results, args.ann, args.iou, args.score)

    return metrics


if __name__ == '__main__':
    main()
