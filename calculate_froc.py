#!/usr/bin/env python3
"""
FROC (Free-Response ROC) calculation for VinBigData chest X-ray detection.

Usage:
    python calculate_froc.py --results results.pkl --ann annotations.json

The VinBigData competition uses mean FROC at FP rates [0.125, 0.25, 0.5, 1, 2, 4, 8].
"""

import argparse
import pickle
import numpy as np
from pycocotools.coco import COCO


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


# Replace your current calculate_froc(...) with this implementation.

def _xyxy_to_xywh(box):
    # box: [x1,y1,x2,y2]
    return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]

def _compute_iou_xywh(b1, b2):
    # both boxes in [x,y,w,h]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    union = (w1 * h1) + (w2 * h2) - inter_area
    return (inter_area / union) if union > 0 else 0.0

def calculate_froc(results_file, ann_file, iou_threshold=0.5):
    import pickle
    from pycocotools.coco import COCO
    from collections import defaultdict

    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)

    print(f"Processing {num_images} images...")

    # build gt structures
    gt_by_image = {}   # img_index -> list of dicts {bbox: [x,y,w,h], cat: raw_cat_id, matched: False}
    total_gt = 0
    for i, img_id in enumerate(img_ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        entries = []
        for a in anns:
            entries.append({'bbox': a['bbox'], 'cat': int(a['category_id']), 'matched': False, 'uid': a.get('id', None)})
            total_gt += 1
        gt_by_image[i] = entries

    # helper: extract per-image predictions as list of (score, box_xyxy, label_raw)
    def extract_preds_for_image(img_res):
        preds = []  # (score, [x1,y1,x2,y2], label_or_-1)
        # Common case: list length==14 where each entry is array of detections for that class
        if isinstance(img_res, (list,tuple)) and len(img_res) == 14:
            for cls_idx, arr in enumerate(img_res):
                a = np.asarray(arr)
                if a.size == 0:
                    continue
                if a.ndim == 1:
                    # single row
                    score = float(a[-1])
                    box = a[:4].astype(float).tolist()
                    preds.append((score, box, int(cls_idx)))
                else:
                    for row in a:
                        row = np.asarray(row)
                        score = float(row[-1])
                        box = row[:4].astype(float).tolist()
                        preds.append((score, box, int(cls_idx)))
            return preds
        # Other formats: Nx6 [x1,y1,x2,y2,score,label]
        try:
            arr = np.asarray(img_res)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                for row in arr:
                    score = float(row[-2])
                    label = int(row[-1])
                    box = row[:4].astype(float).tolist()
                    preds.append((score, box, label))
                return preds
            # Nx5 [x1,y1,x2,y2,score]
            if arr.ndim == 2 and arr.shape[1] == 5:
                for row in arr:
                    score = float(row[-1])
                    box = row[:4].astype(float).tolist()
                    preds.append((score, box, -1))
                return preds
        except Exception:
            pass
        # If we can't parse, return empty
        return preds

    # build a global list of predictions: (score, img_index, xyxy_box, label)
    all_detections = []
    for img_idx, img_res in enumerate(results):
        preds = extract_preds_for_image(img_res)
        for score, box_xyxy, label in preds:
            # Ensure box is [x1,y1,x2,y2]
            # Some code produced [x1,y1,x2,y2] already; we accept it as-is.
            all_detections.append((float(score), int(img_idx), [float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])], int(label)))

    # sort globally by score desc
    all_detections.sort(key=lambda x: x[0], reverse=True)

    # Now iterate predictions and match per-image:
    tp = 0
    fp = 0
    detections_record = []  # (score, is_tp, img_idx)
    # for speed, convert gt_by_image bboxes to xywh arrays
    for img_idx in range(num_images):
        for g in gt_by_image[img_idx]:
            # ensure floats
            g['bbox'] = [float(v) for v in g['bbox']]
            g['matched'] = False

    for score, img_idx, box_xyxy, label in all_detections:
        matched_flag = False
        # find candidate GTs in that image with same class (class-aware)
        gts = gt_by_image.get(img_idx, [])
        best_iou = -1.0
        best_gt_idx = -1
        # convert predicted box to xywh for IoU comparison
        pred_xywh = _xyxy_to_xywh(box_xyxy)
        for gi, g in enumerate(gts):
            if g['matched']:
                continue
            # require class match if pred has a label (>=0); if label==-1 we match any class
            if label >= 0 and int(g['cat']) != int(label):
                continue
            # compute IoU
            iou = _compute_iou_xywh(pred_xywh, g['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        # if best_iou >= iou_threshold -> match
        if best_gt_idx >= 0 and best_iou >= float(iou_threshold):
            gts[best_gt_idx]['matched'] = True
            matched_flag = True
            tp += 1
        else:
            fp += 1
        detections_record.append((score, matched_flag, img_idx))

    # Final counts
    # compute sensitivity curve: iterate detections_record in descending score (already sorted)
    fps_per_image = []
    sensitivities = []
    cur_tp = 0
    cur_fp = 0
    for score, is_tp, img_idx in detections_record:
        if is_tp:
            cur_tp += 1
        else:
            cur_fp += 1
        sensitivities.append(cur_tp / total_gt if total_gt > 0 else 0.0)
        fps_per_image.append(cur_fp / float(num_images))

    # Compute sensitivity at standard FP rates:
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_scores = []
    for fp_rate in fp_rates:
        sens_at = 0.0
        # find the sensitivity at the point where fp_per_img just exceeds fp_rate
        # (or the last sensitivity if we never exceed it)
        for sens, fp_pi in zip(sensitivities, fps_per_image):
            if fp_pi <= fp_rate:
                sens_at = sens
            # Don't break - keep going to find the last valid point
        froc_scores.append(sens_at)
    avg_froc = float(np.mean(froc_scores)) if len(froc_scores) > 0 else 0.0

    # Diagnostic: find how many detections processed at each FP rate
    print("\n" + "="*50)
    print(f"FROC Results (IoU threshold = {iou_threshold})")
    print("="*50)
    print(f"Total ground truth lesions: {total_gt}")
    print(f"Total images: {num_images}")
    print(f"Total detections: {len(all_detections)}")
    print(f"True positives (matched): {tp}")
    print(f"False positives: {fp}")
    print(f"Total Recall (all detections): {tp/total_gt:.4f} ({tp}/{total_gt})")

    # Diagnostic: where are the TPs in the sorted list?
    tp_positions = [i for i, (_, is_tp, _) in enumerate(detections_record) if is_tp]
    if tp_positions:
        print(f"\nDiagnostic - TP positions in sorted detections:")
        print(f"  First TP at position: {tp_positions[0]}")
        print(f"  Last TP at position: {tp_positions[-1]}")
        print(f"  Median TP position: {tp_positions[len(tp_positions)//2]}")
        print(f"  At 8 FP/img ({int(8*num_images)} FPs), detections processed: ~{int(8*num_images) + int(tp * 8 * num_images / fp) if fp > 0 else 0}")
    print("\nSensitivity at FP/image rates:")
    print("-"*30)
    for rate, sens in zip(fp_rates, froc_scores):
        print(f"  {rate:6.3f} FP/img: {sens:.4f}")
    print("-"*30)
    print(f"Average FROC score: {avg_froc:.4f}")
    print("="*50 + "\n")

    return fps_per_image, sensitivities, avg_froc, froc_scores




def plot_froc(fps_per_image, sensitivities, output_file='froc_curve.png', froc_scores=None, avg_froc=None):
    """Plot and save FROC curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the FROC curve
        ax.plot(fps_per_image, sensitivities, 'b-', linewidth=2.5, label='FROC Curve')

        # Mark standard FP rates and sensitivity points
        fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        for i, fp_rate in enumerate(fp_rates):
            ax.axvline(x=fp_rate, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            # Add FP rate labels at top
            ax.text(fp_rate, 1.02, f'{fp_rate}', ha='center', va='bottom', fontsize=9, color='gray')

        # Mark sensitivity points at each FP rate if scores provided
        if froc_scores:
            for fp_rate, sens in zip(fp_rates, froc_scores):
                ax.plot(fp_rate, sens, 'ro', markersize=8)
                ax.text(fp_rate + 0.2, sens, f'{sens:.3f}', fontsize=9, va='center')

        ax.set_xlabel('Average False Positives per Image', fontsize=14)
        ax.set_ylabel('Sensitivity (Recall)', fontsize=14)

        # Title with FROC score
        title = 'FROC Curve - VinBigData Chest X-ray Detection'
        if avg_froc is not None:
            title += f'\nMean FROC: {avg_froc:.4f}'
        ax.set_title(title, fontsize=16)

        ax.grid(True, alpha=0.3)
        # Limit X-axis to 0-10 to focus on the important region
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1.05])

        # Add legend
        if froc_scores:
            ax.plot([], [], 'ro', markersize=8, label='Sensitivity at FP rates')
            ax.legend(loc='lower right', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"FROC curve saved to {output_file}")

    except ImportError:
        print("matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(description='Calculate FROC for VinBigData detection')
    parser.add_argument('--results', type=str,
                        default='work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results.pkl',
                        help='Path to results.pkl from test.py')
    parser.add_argument('--ann', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations/instances_val.json',
                        help='Path to COCO format annotations')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--plot', type=str, default='froc_curve.png',
                        help='Output path for FROC curve plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')

    args = parser.parse_args()

    # Calculate FROC
    fps, sens, avg_froc, froc_scores = calculate_froc(
        args.results, args.ann, args.iou
    )

    # Plot FROC curve
    if not args.no_plot and len(fps) > 0:
        plot_froc(fps, sens, args.plot, froc_scores, avg_froc)

    return avg_froc


if __name__ == '__main__':
    main()
