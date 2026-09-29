"""metrics.py: a hand-computed classification example, and detection without predictions."""
import json
from pathlib import Path

import pytest

from metrics import classification_metrics, detection_metrics

ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'SARFact-Course-20K'


def test_classification_metrics_by_hand():
    metrics = classification_metrics([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2], num_classes=3)
    assert metrics['confusion_matrix'] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert metrics['accuracy'] == pytest.approx(4 / 6)
    assert metrics['balanced_accuracy'] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)  # recall per class
    # precision 1, 1/2, 1/2 and recall 2/3, 1/2, 1 give F1 = 4/5, 1/2, 2/3
    assert metrics['macro_f1'] == pytest.approx((4 / 5 + 1 / 2 + 2 / 3) / 3)


def test_detection_metrics_without_predictions_is_zero():
    gt_json = ROOT / 'detection' / 'instances_val.json'
    image_ids = [image['id'] for image in json.loads(gt_json.read_text())['images']]
    metrics = detection_metrics(gt_json, [], image_ids)
    assert metrics['mAP'] == metrics['AP50'] == metrics['AP75'] == 0
    assert list(metrics['per_class_AP'].values()) == [0] * 9
