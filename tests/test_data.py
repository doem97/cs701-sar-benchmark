"""data.py on the real dataset: split contents, the box geometry round trip, and flips."""
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from data import MEAN, STD, SARMultiTask, collate, to_original_xywh
from metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'SARFact-Course-20K'


def batches(split):
    return DataLoader(SARMultiTask(ROOT, split), batch_size=64, collate_fn=collate)


@pytest.mark.parametrize('split, size', [('train', 9392), ('val', 1426), ('test', 2065)])
def test_subset_a(split, size):
    loader = batches(split)
    assert len(loader.dataset) == size
    for images, labels, targets in loader:
        assert images.shape[1:] == (1, 512, 512) and images.dtype == torch.float32
        for label, target in zip(labels.tolist(), targets):
            assert len(target['labels']) >= 1
            assert (target['labels'] == label + 1).all()
            assert target['boxes'].min() >= 0 and target['boxes'].max() <= 512


@pytest.mark.parametrize('split', ['val', 'test'])
def test_ground_truth_round_trip_scores_map_1(split):
    predictions, image_ids = [], []
    for _, _, targets in batches(split):
        for t in targets:
            image_ids.append(t['image_id'])
            boxes = to_original_xywh(t['boxes'], t['orig_size']).tolist()
            predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': 1.0}
                            for b, c in zip(boxes, t['labels'].tolist())]
    metrics = detection_metrics(ROOT / 'detection' / f'instances_{split}.json', predictions, image_ids)
    assert round(metrics['mAP'], 3) == 1.0
    assert [round(ap, 3) for ap in metrics['per_class_AP'].values()] == [1.0] * 9


def test_flips_move_boxes_with_pixels():
    transform = SARMultiTask(ROOT, 'train', train=True).transform
    image = torch.zeros(1, 200, 300, dtype=torch.uint8)  # non-square, so the resize also stretches it
    image[:, 50:90, 30:120] = 255  # a bright rectangle filling exactly this box:
    box = tv_tensors.BoundingBoxes([[30., 50., 120., 90.]], format='XYXY', canvas_size=(200, 300))
    flips = set()
    for seed in range(8):
        torch.manual_seed(seed)
        out, out_box = transform(image, box)
        ys, xs = torch.nonzero(out[0] * STD + MEAN > 0.5, as_tuple=True)
        bright = torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])  # pixel i spans [i, i + 1)
        assert torch.allclose(bright.float(), out_box[0], atol=1)
        flips.add((out_box[0, 0].item() > 256, out_box[0, 1].item() > 256))  # (horizontal, vertical)
    assert len(flips) == 4  # the seeds cover every combination of the two flips
