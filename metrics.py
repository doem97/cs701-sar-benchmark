"""Evaluation metrics: classification scores from a confusion matrix, and COCO box AP via pycocotools."""
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def classification_metrics(y_true, y_pred, num_classes=9):
    """Accuracy, macro-F1, balanced accuracy (mean per-class recall) and the confusion matrix
    (rows = truth, columns = prediction). Every class must occur in y_true, as in every split."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    recall = tp / support
    f1 = 2 * tp / (support + predicted)  # = 2 TP / (2 TP + FN + FP), defined for never-predicted classes
    return {
        'accuracy': float(tp.sum() / cm.sum()),
        'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall.mean()),
        'confusion_matrix': cm.tolist(),
    }


def detection_metrics(gt_json, predictions, image_ids):
    """COCO box AP of `predictions` (dicts with image_id, category_id, bbox [x, y, w, h] in original
    pixels, score) against the ground truth in `gt_json`, evaluated on the images `image_ids` only."""
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools; the caller prints its own summary
        gt = COCO(gt_json)
        # loadRes adds keys to the dicts it is given, so it gets copies; it cannot load an empty list
        dt = gt.loadRes([dict(p) for p in predictions]) if predictions else COCO()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval['precision'][:, :, :, 0, -1]  # [IoU threshold, recall, class] at area "all", 100 dets
    return {
        'mAP': float(ev.stats[0]),
        'AP50': float(ev.stats[1]),
        'AP75': float(ev.stats[2]),
        'per_class_AP': {gt.cats[c]['name']: float(precision[:, :, k].mean())  # -1: no boxes of that class
                         for k, c in enumerate(ev.params.catIds)},
    }
