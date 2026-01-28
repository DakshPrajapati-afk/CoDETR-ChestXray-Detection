#!/usr/bin/env python3
"""
FROC calculation with multiple diseases plotted together in VinBigData.

Creates 4 plots, each showing 3-4 individual disease FROC curves together:
1. Plot 1: Aortic enlargement, Cardiomegaly, Pleural effusion, Pneumothorax
2. Plot 2: Atelectasis, Consolidation, ILD, Infiltration
3. Plot 3: Lung Opacity, Nodule/Mass, Other lesion
4. Plot 4: Calcification, Pleural thickening, Pulmonary fibrosis

Usage:
    python calculate_froc_grouped.py --results results.pkl --ann annotations.json
"""

import argparse
import pickle
import numpy as np
import os
from pycocotools.coco import COCO

# VinBigData class names (0-indexed)
CLASS_NAMES = [
    'Aortic enlargement',    # 0
    'Atelectasis',           # 1
    'Calcification',         # 2
    'Cardiomegaly',          # 3
    'Consolidation',         # 4
    'ILD',                   # 5
    'Infiltration',          # 6
    'Lung Opacity',          # 7
    'Nodule/Mass',           # 8
    'Other lesion',          # 9
    'Pleural effusion',      # 10
    'Pleural thickening',    # 11
    'Pneumothorax',          # 12
    'Pulmonary fibrosis'     # 13
]

# Disease groupings for plotting (3-4 diseases per plot)
PLOT_GROUPS = {
    'Plot 1 - Cardiac & Pleural': [0, 3, 10, 12],  # Aortic enlargement, Cardiomegaly, Pleural effusion, Pneumothorax
    'Plot 2 - Lung Infections': [1, 4, 5, 6],  # Atelectasis, Consolidation, ILD, Infiltration
    'Plot 3 - Opacities & Masses': [7, 8, 9],  # Lung Opacity, Nodule/Mass, Other lesion
    'Plot 4 - Chronic Conditions': [2, 11, 13]  # Calcification, Pleural thickening, Pulmonary fibrosis
}

# Colors for individual diseases (distinct colors)
DISEASE_COLORS = [
    '#E74C3C',  # Red - Aortic enlargement
    '#3498DB',  # Blue - Atelectasis
    '#2ECC71',  # Green - Calcification
    '#9B59B6',  # Purple - Cardiomegaly
    '#F39C12',  # Orange - Consolidation
    '#1ABC9C',  # Teal - ILD
    '#E91E63',  # Pink - Infiltration
    '#00BCD4',  # Cyan - Lung Opacity
    '#8BC34A',  # Light Green - Nodule/Mass
    '#FF5722',  # Deep Orange - Other lesion
    '#673AB7',  # Deep Purple - Pleural effusion
    '#795548',  # Brown - Pleural thickening
    '#607D8B',  # Blue Grey - Pneumothorax
    '#CDDC39'   # Lime - Pulmonary fibrosis
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


def calculate_froc_for_group(results, coco, img_ids, class_ids, iou_threshold=0.5):
    """
    Calculate FROC curve for a group of classes.

    Args:
        class_ids: List of class IDs to include in this group
    """
    num_images = len(img_ids)

    # Collect all detections for this group
    all_detections = []
    total_gt = 0

    for idx, img_id in enumerate(img_ids):
        # Get ground truth for classes in this group
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_anns = coco.loadAnns(ann_ids)

        # Filter GT for classes in this group
        gt_boxes = [ann['bbox'] for ann in gt_anns if ann['category_id'] in class_ids]
        gt_matched = [False] * len(gt_boxes)
        total_gt += len(gt_boxes)

        # Get predictions for classes in this group
        all_preds = []
        for cls_id in class_ids:
            for pred in results[idx][cls_id]:
                all_preds.append((pred[4], pred[:4]))

        # Sort by score (descending)
        all_preds.sort(key=lambda x: x[0], reverse=True)

        for score, pred_box in all_preds:
            pred_xywh = [pred_box[0], pred_box[1], pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]]

            is_tp = False
            best_iou = iou_threshold
            best_gt_idx = -1

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

    # Sort all detections by score
    all_detections.sort(key=lambda x: x[0], reverse=True)

    # Calculate FROC curve
    fps_per_image = []
    sensitivities = []

    tp = 0
    fp = 0

    for score, is_tp in all_detections:
        if is_tp:
            tp += 1
        else:
            fp += 1

        sensitivity = tp / total_gt if total_gt > 0 else 0
        fp_per_img = fp / num_images

        sensitivities.append(sensitivity)
        fps_per_image.append(fp_per_img)

    # Calculate average FROC score at standard FP rates
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_scores = []

    for fp_rate in fp_rates:
        sens = 0
        for i, fp_per_img in enumerate(fps_per_image):
            if fp_per_img <= fp_rate:
                sens = sensitivities[i]
        froc_scores.append(sens)

    avg_froc = np.mean(froc_scores)

    return fps_per_image, sensitivities, avg_froc, total_gt, froc_scores


def calculate_froc_for_single_class(results, coco, img_ids, class_id, iou_threshold=0.5):
    """Calculate FROC curve for a single class."""
    return calculate_froc_for_group(results, coco, img_ids, [class_id], iou_threshold)


def calculate_grouped_frocs(results_file, ann_file, iou_threshold=0.5, output_dir='froc_grouped'):
    """Calculate FROC for each disease and generate multi-disease plots."""

    # Load data
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)

    print(f"Processing {num_images} images...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Store results for all individual diseases
    disease_results = {}
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    # Calculate FROC for each individual disease
    print("\n" + "="*70)
    print("FROC RESULTS PER DISEASE")
    print("="*70)
    print(f"{'Disease':<25} {'GT Count':>10} {'Mean FROC':>12}")
    print("-"*70)

    for class_id in range(len(CLASS_NAMES)):
        fps, sens, avg_froc, total_gt, froc_scores = calculate_froc_for_single_class(
            results, coco, img_ids, class_id, iou_threshold
        )

        disease_results[class_id] = {
            'name': CLASS_NAMES[class_id],
            'fps': fps,
            'sens': sens,
            'avg_froc': avg_froc,
            'total_gt': total_gt,
            'froc_scores': froc_scores
        }

        print(f"{CLASS_NAMES[class_id]:<25} {total_gt:>10} {avg_froc:>12.4f}")

    print("-"*70)
    overall_froc = np.mean([r['avg_froc'] for r in disease_results.values()])
    print(f"{'Overall Mean':<25} {'':<10} {overall_froc:>12.4f}")
    print("="*70)

    # Detailed breakdown
    print("\nDETAILED FROC AT EACH FP RATE")
    print("="*70)

    header = f"{'Disease':<20}"
    for fp_rate in fp_rates:
        header += f" {fp_rate:>7}"
    header += f" {'Mean':>8}"
    print(header)
    print("-"*70)

    for class_id in range(len(CLASS_NAMES)):
        r = disease_results[class_id]
        line = f"{r['name'][:19]:<20}"
        for score in r['froc_scores']:
            line += f" {score:>7.3f}"
        line += f" {r['avg_froc']:>8.4f}"
        print(line)

    print("="*70)

    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Generate grouped plots (multiple diseases per plot)
        print(f"\nGenerating grouped FROC plots (multiple diseases per plot)...")

        for group_name, class_ids in PLOT_GROUPS.items():
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot each disease in this group
            for class_id in class_ids:
                r = disease_results[class_id]
                if len(r['fps']) > 0:
                    ax.plot(r['fps'], r['sens'],
                           color=DISEASE_COLORS[class_id],
                           linewidth=2.5,
                           label=f"{r['name']} (FROC={r['avg_froc']:.3f}, n={r['total_gt']})")

            # Mark standard FP rates
            for fp_rate in fp_rates:
                ax.axvline(x=fp_rate, color='gray', linestyle='--', alpha=0.4, linewidth=1)

            # Calculate group average FROC
            group_avg_froc = np.mean([disease_results[cid]['avg_froc'] for cid in class_ids])

            ax.set_xlabel('Average False Positives per Image', fontsize=14)
            ax.set_ylabel('Sensitivity (Recall)', fontsize=14)
            ax.set_title(f'{group_name}\nGroup Mean FROC: {group_avg_froc:.4f}', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 10])
            ax.set_ylim([0, 1])
            ax.legend(loc='lower right', fontsize=11)

            plt.tight_layout()
            filename = f"{output_dir}/froc_{group_name.replace(' ', '_').replace('-', '').replace('  ', '_')}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"  Saved: {filename}")

        # Combined plot with all 14 diseases
        print(f"\nGenerating combined FROC plot (all diseases)...")

        fig, ax = plt.subplots(figsize=(14, 10))

        for class_id in range(len(CLASS_NAMES)):
            r = disease_results[class_id]
            if len(r['fps']) > 0:
                ax.plot(r['fps'], r['sens'],
                       color=DISEASE_COLORS[class_id],
                       linewidth=1.5,
                       label=f"{r['name']} ({r['avg_froc']:.3f})")

        # Mark standard FP rates
        for fp_rate in fp_rates:
            ax.axvline(x=fp_rate, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Average False Positives per Image', fontsize=14)
        ax.set_ylabel('Sensitivity (Recall)', fontsize=14)
        ax.set_title(f'FROC Curves - All 14 Diseases\nOverall Mean FROC: {overall_froc:.4f}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        filename = f"{output_dir}/froc_all_diseases_combined.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

        # Bar chart comparison
        print(f"\nGenerating comparison bar chart...")

        fig, ax = plt.subplots(figsize=(14, 8))

        class_names = [r['name'] for r in disease_results.values()]
        froc_values = [r['avg_froc'] for r in disease_results.values()]
        gt_counts = [r['total_gt'] for r in disease_results.values()]
        colors = DISEASE_COLORS

        x = np.arange(len(class_names))
        bars = ax.bar(x, froc_values, color=colors, edgecolor='black', linewidth=1)

        # Add GT count labels on bars
        for bar, gt in zip(bars, gt_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'n={gt}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.axhline(y=overall_froc, color='red', linestyle='--', linewidth=2,
                  label=f'Overall Mean: {overall_froc:.3f}')

        ax.set_xlabel('Disease', fontsize=12)
        ax.set_ylabel('Mean FROC Score', fontsize=12)
        ax.set_title('Mean FROC Score by Disease', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f"{output_dir}/froc_bar_chart.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

        print(f"\nAll plots saved to: {output_dir}/")

    except ImportError:
        print("\nmatplotlib not available, skipping plot generation")

    return disease_results


def main():
    parser = argparse.ArgumentParser(description='Calculate FROC for grouped disease categories')
    parser.add_argument('--results', type=str,
                        default='work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results_ep44.pkl',
                        help='Path to results.pkl')
    parser.add_argument('--ann', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations/instances_val.json',
                        help='Path to annotations')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--output', type=str, default='froc_grouped',
                        help='Output directory for plots')

    args = parser.parse_args()

    results = calculate_grouped_frocs(args.results, args.ann, args.iou, args.output)

    return results


if __name__ == '__main__':
    main()
