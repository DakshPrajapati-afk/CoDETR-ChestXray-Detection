#!/usr/bin/env python3
"""
AUROC (Area Under the ROC Curve) calculation for VinBigData chest X-ray detection.

This calculates AUROC at the image level - whether each disease is present or absent in an image.

Usage:
    python calculate_auroc.py --results results.pkl --ann annotations.json
"""

import argparse
import pickle
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
    'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
    'Pneumothorax', 'Pulmonary fibrosis'
]


def calculate_auroc(results_file, ann_file, score_threshold=0.0):
    """
    Robust AUROC calculation + automatic diagnostics for unexpected results.
    """
    import random
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    num_classes = len(CLASS_NAMES)

    print(f"Processing {num_images} images for {num_classes} classes...")

    # Quick summary of results structure (diagnostic)
    try:
        print("=== RESULTS STRUCTURE DIAGNOSTIC ===")
        print("results type:", type(results), "len:", len(results) if hasattr(results, '__len__') else 'N/A')
        x0 = results[0]
        print("first elem type:", type(x0))
        try:
            import numpy as _np
            x0_arr = _np.asarray(x0)
            print("first elem array shape:", x0_arr.shape, "dtype:", x0_arr.dtype)
            if x0_arr.ndim == 2:
                print("  (2D) first 3 rows:\n", x0_arr[:3])
            elif x0_arr.ndim == 1:
                print("  (1D) first 10 values:", x0_arr[:10])
        except Exception:
            pass
        if isinstance(x0, (list, tuple)):
            print("first 3 entries types/shapes:", [(type(e), (len(e) if hasattr(e,'__len__') else None)) for e in x0[:3]])
        if isinstance(x0, dict):
            print("dict keys:", list(x0.keys()))
            for k in list(x0.keys())[:5]:
                v = x0[k]
                print(f"  key={k} type={type(v)} shape={(len(v) if hasattr(v,'__len__') else None)} sample={(v[:3] if hasattr(v,'__len__') else v)}")
        print("=== END RESULTS DIAGNOSTIC ===\n")
    except Exception as e:
        print("Could not print results diagnostic:", e)

    # Build mapping from COCO category_id -> class index (0..num_classes-1)
    cat_ids = coco.getCatIds()
    coco_cats = coco.loadCats(cat_ids)
    catid_to_idx = {}
    for cat in coco_cats:
        cid = cat['id']
        name = cat.get('name', '')
        if name in CLASS_NAMES:
            catid_to_idx[cid] = CLASS_NAMES.index(name)
        else:
            if 0 <= cid - 1 < num_classes:
                catid_to_idx[cid] = cid - 1

    # Prepare arrays
    gt_labels = np.zeros((num_images, num_classes), dtype=np.int32)
    pred_scores = np.zeros((num_images, num_classes), dtype=np.float32)

    # map img id -> index
    imgid_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

    # Build GT
    for img_id in img_ids:
        idx = imgid_to_idx[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_anns = coco.loadAnns(ann_ids)
        for ann in gt_anns:
            ann_cat_id = ann['category_id']
            if ann_cat_id in catid_to_idx:
                cls_idx = catid_to_idx[ann_cat_id]
                gt_labels[idx, cls_idx] = 1

    # Detect results container type
    results_is_dict = isinstance(results, dict)
    if results_is_dict:
        print("Results: dict keyed by image id detected.")
    else:
        print("Results: sequence detected (list/tuple).")

    def get_img_results(img_id, idx):
        if results_is_dict:
            if img_id in results:
                return results[img_id]
            if str(img_id) in results:
                return results[str(img_id)]
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info.get('file_name', None)
            if file_name and file_name in results:
                return results[file_name]
            return None
        else:
            if idx < len(results):
                return results[idx]
            return None

    # Auto-detection and parsing per-image (same robust parser as before)
    for idx, img_id in enumerate(img_ids):
        img_res = get_img_results(img_id, idx)
        if img_res is None:
            continue

        # Case A: per-class list (len == num_classes)
        if (isinstance(img_res, (list, tuple))
                and len(img_res) == num_classes
                and (len(img_res) == 0 or hasattr(img_res[0], '__len__'))):
            for cls_id, class_preds in enumerate(img_res):
                if class_preds is None:
                    continue
                arr = np.asarray(class_preds)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    score = float(arr[-1])
                    if score >= score_threshold:
                        pred_scores[idx, cls_id] = max(pred_scores[idx, cls_id], score)
                else:
                    scores = arr[:, -1].astype(float)
                    valid = scores[scores >= score_threshold]
                    if valid.size > 0:
                        pred_scores[idx, cls_id] = max(pred_scores[idx, cls_id], float(np.max(valid)))
            continue

        # Case B: dict with scores+labels
        if isinstance(img_res, dict):
            scores = None
            labels = None
            for k in ['scores', 'score', 'conf', 'confidences']:
                if k in img_res:
                    scores = np.asarray(img_res[k]).astype(float)
                    break
            for k in ['labels', 'classes', 'class_ids', 'category_ids']:
                if k in img_res:
                    labels = np.asarray(img_res[k]).astype(int)
                    break
            if scores is not None and labels is not None:
                for s, l in zip(scores, labels):
                    if s < score_threshold:
                        continue
                    l0 = int(l)
                    if 0 <= l0 < num_classes:
                        cls_idx = l0
                    elif 1 <= l0 <= num_classes:
                        cls_idx = l0 - 1
                    else:
                        continue
                    pred_scores[idx, cls_idx] = max(pred_scores[idx, cls_idx], float(s))
                continue

        # Case C: array-like per-image NxK
        arr = np.asarray(img_res)
        if arr.ndim == 1:
            if arr.size >= 6:
                score = float(arr[-2])
                label = int(arr[-1])
                if score >= score_threshold:
                    if 0 <= label < num_classes:
                        cls_idx = label
                    elif 1 <= label <= num_classes:
                        cls_idx = label - 1
                    else:
                        continue
                    pred_scores[idx, cls_idx] = max(pred_scores[idx, cls_idx], score)
            continue

        if arr.ndim == 2:
            ncols = arr.shape[1]
            if ncols >= 6:
                scores = arr[:, -2].astype(float)
                labels = arr[:, -1].astype(int)
                for s, l in zip(scores, labels):
                    if s < score_threshold:
                        continue
                    l0 = int(l)
                    if 0 <= l0 < num_classes:
                        cls_idx = l0
                    elif 1 <= l0 <= num_classes:
                        cls_idx = l0 - 1
                    else:
                        continue
                    pred_scores[idx, cls_idx] = max(pred_scores[idx, cls_idx], float(s))
                continue
            else:
                # ncols == 5 -> only scores, cannot map to classes -> skip
                continue

    # --- Diagnostics: per-class statistics (mean pos/neg, fraction zeros) ---
    print("\n=== PER-CLASS SCORE DIAGNOSTICS ===")
    per_class_stats = []
    for cid, cname in enumerate(CLASS_NAMES):
        y_true = gt_labels[:, cid]
        y_score = pred_scores[:, cid]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            per_class_stats.append((cname, n_pos, n_neg, np.nan, np.nan, np.nan, np.nan))
            continue
        mean_pos = float(y_score[y_true==1].mean()) if n_pos>0 else float('nan')
        mean_neg = float(y_score[y_true==0].mean()) if n_neg>0 else float('nan')
        frac_pos_zero = float((y_score[y_true==1] == 0).sum()) / max(1, n_pos)
        frac_neg_zero = float((y_score[y_true==0] == 0).sum()) / max(1, n_neg)
        const_pos = np.allclose(y_score[y_true==1], y_score[y_true==1][0]) if n_pos>0 else False
        const_neg = np.allclose(y_score[y_true==0], y_score[y_true==0][0]) if n_neg>0 else False
        per_class_stats.append((cname, n_pos, n_neg, mean_pos, mean_neg, frac_pos_zero, frac_neg_zero, const_pos, const_neg))

    print(f"{'Class':<25} {'pos':>4} {'neg':>4} {'mean_pos':>9} {'mean_neg':>9} {'pos_zero':>9} {'neg_zero':>9} {'const_pos':>10} {'const_neg':>10}")
    for row in per_class_stats:
        cname = row[0]
        if np.isnan(row[3]):
            print(f"{cname:<25} {row[1]:4d} {row[2]:4d} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{cname:<25} {row[1]:4d} {row[2]:4d} {row[3]:9.4f} {row[4]:9.4f} {row[5]:9.3f} {row[6]:9.3f} {str(row[7]):>10} {str(row[8]):>10}")
    print("=== END PER-CLASS DIAGNOSTICS ===\n")

    # sample image checks
    rand_samples = random.sample(range(num_images), min(5, num_images))
    print("Sample image checks (img_idx, img_id, top_pred_class(top_score), GT positive classes):")
    for i in rand_samples:
        img_id = img_ids[i]
        top_cls = int(np.argmax(pred_scores[i]))
        top_score = float(pred_scores[i, top_cls])
        true_classes = [CLASS_NAMES[j] for j in np.where(gt_labels[i] == 1)[0].tolist()]
        print(f"  idx={i} img_id={img_id}  top_pred=({top_cls}:{CLASS_NAMES[top_cls]}, {top_score:.3f})  GT={true_classes}")

    # Compute AUROC results (unchanged)
    auroc_per_class = {}
    valid_aurocs = []
    for cls_id in range(num_classes):
        y_true = gt_labels[:, cls_id]
        y_score = pred_scores[:, cls_id]
        n_pos = int(np.sum(y_true))
        n_neg = int(len(y_true) - n_pos)
        if n_pos > 0 and n_neg > 0:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except ValueError as e:
                print(f"Warning computing AUROC for class {cls_id}: {e}")
                auroc = None
            auroc_per_class[cls_id] = auroc
            if auroc is not None:
                valid_aurocs.append(auroc)
        else:
            auroc_per_class[cls_id] = None

    mean_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0

    # Print results (same format)
    print(f"\n{'='*60}")
    print(f"AUROC Results (Image-Level Classification)")
    print(f"{'='*60}")
    print(f"\n{'Class':<25} {'AUROC':>10} {'Pos Images':>12} {'Neg Images':>12}")
    print(f"{'-'*60}")
    for cls_id in range(num_classes):
        y_true = gt_labels[:, cls_id]
        n_pos = int(np.sum(y_true))
        n_neg = int(len(y_true) - n_pos)
        auroc = auroc_per_class[cls_id]
        if auroc is not None:
            print(f"{CLASS_NAMES[cls_id]:<25} {auroc:>10.4f} {n_pos:>12} {n_neg:>12}")
        else:
            print(f"{CLASS_NAMES[cls_id]:<25} {'N/A':>10} {n_pos:>12} {n_neg:>12}")
    print(f"{'-'*60}")
    print(f"{'Mean AUROC':<25} {mean_auroc:>10.4f}")
    print(f"{'='*60}\n")

    return auroc_per_class, mean_auroc, gt_labels, pred_scores




def plot_roc_curves(gt_labels, pred_scores, output_file='auroc_curves.png'):
    """Plot ROC curves for all classes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab20(np.linspace(0, 1, len(CLASS_NAMES)))

        aurocs = []
        for cls_id in range(len(CLASS_NAMES)):
            y_true = gt_labels[:, cls_id]
            y_score = pred_scores[:, cls_id]

            n_pos = np.sum(y_true)
            n_neg = len(y_true) - n_pos

            if n_pos > 0 and n_neg > 0:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auroc = roc_auc_score(y_true, y_score)
                aurocs.append(auroc)

                ax.plot(fpr, tpr, color=colors[cls_id], linewidth=1.5,
                       label=f'{CLASS_NAMES[cls_id]} ({auroc:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

        mean_auroc = np.mean(aurocs) if aurocs else 0.0

        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f'ROC Curves - VinBigData Chest X-ray Detection\nMean AUROC: {mean_auroc:.4f}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to {output_file}")

    except ImportError as e:
        print(f"matplotlib or sklearn not available: {e}")


def main():
    parser = argparse.ArgumentParser(description='Calculate AUROC for VinBigData detection')
    parser.add_argument('--results', type=str,
                        default='work_dirs/co_dino_5scale_r50_vinbigdata_dicom/results.pkl',
                        help='Path to results.pkl from test.py')
    parser.add_argument('--ann', type=str,
                        default='/scratch/dpraja12/data/VinBigData/annotations/instances_val.json',
                        help='Path to COCO format annotations')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Score threshold for detections (default: 0.0)')
    parser.add_argument('--plot', type=str, default='auroc_curves.png',
                        help='Output path for ROC curves plot')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')

    args = parser.parse_args()

    # Calculate AUROC
    auroc_per_class, mean_auroc, gt_labels, pred_scores = calculate_auroc(
        args.results, args.ann, args.threshold
    )

    # Plot ROC curves
    if not args.no_plot:
        plot_roc_curves(gt_labels, pred_scores, args.plot)

    return mean_auroc


if __name__ == '__main__':
    main()
