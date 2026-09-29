# CS701 SAR multi-task benchmark

A small PyTorch code base that trains one ViT-B/16 backbone jointly for **image classification**
(9 classes) and **object detection** (Faster R-CNN) on the CS701 SAR course data. It reproduces eight
reference baselines: two pretrained backbones, each adapted in three ways, plus both architectures
trained from scratch.

## Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench && pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 builds of torch 2.9.0 and torchvision 0.24.0. The pretrained
weights are downloaded from Hugging Face on first use (ViT: 0.4 GB, TerraMind: 1.5 GB).

## Data

The course data is described on its dataset card: https://huggingface.co/datasets/doem1997/cs701-sar-course-data.
Every image has one class label and at least one box, all of that class, so both tasks are learned and
scored on the same images. You get labels and boxes for train only; Codabench scores your val predictions
in Phase 1 and your test predictions in Phase 2.

The course staff ran this code on the full labelled data (val and test annotations included), stored in a
different folder layout: `train.py --data PATH` expects `classification/{split}.csv` and
`detection/instances_{split}.json` for train, val and test, and scores val and test itself. To use the code
on the course data, adapt the data loading in `data.py` and `train.py`:

- train on `train/labels.csv` and `train/instances.json`; predict on the images of `val/images.json` and
  `test/images.json`, which have no annotations;
- hold out part of train for local validation;
- write the submission zip, `classification.csv` and `detection.json` (format on the dataset card), with boxes
  in original image pixels, as `data.to_original_xywh` already gives them. `make_submission.py` has a short
  `write_submission` function that writes the zip from your predictions (the one shown in the briefing video).

## Run

```bash
python train.py --backbone vit --init pretrained --adapt lora   # one experiment
bash run_all.sh                                                  # all eight, one after another
```

`--backbone vit|terramind`, `--init pretrained|scratch`, `--adapt full|lora|moelora`; `python train.py -h`
lists the other options. A pretrained `full` run needs `--backbone-lr 2e-5`, as in `run_all.sh`, to match
the reference protocol. `bash run_all.sh --seed 1 --out runs/seed1` repeats all eight with another seed
(the results below are over seeds 0, 1 and 2).

A run alone on one RTX PRO 6000 Blackwell GPU takes about 50 min (LoRA about 47 min, MoE-LoRA about 1 h), and
all eight in sequence about 7 h. A run keeps about 11-17 GiB of GPU memory in use; TerraMind with MoE-LoRA
needs the most and does not fit on a 16 GB card. Lowering `--batch-size` saves memory but changes the
protocol.

A run writes `runs/<backbone>_<init>_<adapt>/`:

- `log.txt`: training progress and the validation and test scores
- `metrics.json`: arguments, parameter counts, `train_hours` (wall time of the epoch loop, including the
  periodic validation; it grows if the GPU is shared), `peak_gpu_memory_gb` (`torch.cuda.max_memory_allocated`
  in GiB; the process needs about 2.5 GiB more), every validation result, and the final validation and
  test metrics
- `predictions_val.json`, `predictions_test.json`: detections in COCO result format (original pixels)
- `model.pt`: the final weights (`state_dict`)

Checks: `bash run_all.sh --limit 64 --epochs 1 --out runs/smoke` runs all eight configurations end to
end on 64 images per split in a few minutes, and `python -m pytest tests` runs the tests (needs a GPU).
`test_data.py`, `test_metrics.py` and `test_model.py` read the staff data layout described above, so they
need the same adaptation; `test_backbones.py` and `test_adapters.py` need no data.

## Files

| file | role |
|---|---|
| `data.py` | images and boxes at 512 x 512, flips, normalization; boxes back to original pixels |
| `backbones.py` | the two ViT-B/16 backbones: timm ViT (ImageNet-21k) and TerraMind-1.0-base |
| `adapters.py` | LoRA and MoE-LoRA around the attention layers of a frozen backbone |
| `model.py` | backbone + classification head + ViTDet feature pyramid + Faster R-CNN |
| `metrics.py` | accuracy, macro-F1, balanced accuracy, confusion matrix; COCO box AP |
| `train.py` | one experiment: train, validate, test, save |
| `run_all.sh` | the eight reference runs |
| `make_submission.py` | writes a Codabench submission zip from predictions |
| `tests/` | checks of the data, metrics, backbones, adapters and model |

## Model

```
image [B,1,512,512] -> backbone -> tokens [B,1024,768]   (32 x 32 patches of 16 px)
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens as a [B,768,32,32] map -> SimpleFeaturePyramid (strides 4 to 64, 256 channels)
                  -> RegionProposalNetwork + RoIHeads (torchvision Faster R-CNN, 9 classes + background)
loss = cross-entropy + the four Faster R-CNN losses
```

Backbones (`--backbone`): `vit` is timm's `vit_base_patch16_224.augreg_in21k` (ImageNet-21k); `terramind`
is TerraMind-1.0-base (IBM/ESA), an Earth-observation model, through its Sentinel-1 GRD input. Adaptation
(`--adapt`):

- `full`: every backbone weight trains.
- `lora`: the backbone is frozen; the `qkv` and `proj` layers of every attention block get
  `y = W x + (alpha / r) B A x` with r = 16, alpha = 32 (0.88 M trainable backbone parameters).
- `moelora`: as `lora`, but the rank-16 update is split into 4 experts of rank 4 with alpha = 8 (so
  alpha / r = 2, as in `lora`), mixed per token by a softmax router; each task has its own router
  (1.03 M trainable backbone parameters).

## Training protocol (the same for all eight runs)

AdamW with weight decay 0.05, batch size 16, bf16 autocast. Learning rate 1e-4, except 2e-5 for the
backbone in the two pretrained `full` runs (`--backbone-lr 2e-5`, as in `run_all.sh`: a pretrained
backbone is fine-tuned gently). One epoch of linear warmup, then cosine decay to 0, updated every
iteration; 24 epochs. Validation every 4 epochs and after the last one; test once, at the end. The
reported model is the last-epoch model: there is no checkpoint selection.

## Design notes

- **Both labels on every image.** Every image has a class label and boxes, so every training batch trains
  both heads.
- **512 px input.** Every image is resized to 512 x 512 (non-square images are stretched), giving
  1,024 tokens. Objects are small (median box about 16 px at 512 px), so the smallest anchors are 8 px on the
  stride-4 level. Predicted boxes are mapped back to original pixels before COCO scoring.
- **Normalization.** The images are gray and are read as one channel. Both backbones get the same input
  normalization: one mean and std (0.2526, 0.2133), computed over the 9,392 train images at 512 px.
- **Precision rule.** The backbone, neck and classification head run in bf16; the RPN and RoI heads run
  in fp32. torchvision's box coder casts anchors to the dtype of the regression output, and bf16 numbers
  between 256 and 512 are 2 px apart, too coarse for objects of 6-15 px. Importing terratorch
  (`backbones.py` does, for every run) switches fp32 matrix multiplications on the GPU to TF32 for the
  whole process; box coordinates are decoded elementwise in fp32 and are unaffected.
- **MoE-LoRA runs the backbone twice.** Its routers are task-specific, so classification and detection
  see different features, and every forward pass runs the backbone once per task.
- **TerraMind domain gap.** TerraMind was pretrained on Sentinel-1 GRD backscatter in dB (two
  polarizations, VV and VH, 10 m pixels). Our images are 8-bit gray display images, mostly at a higher
  resolution; the gray channel is fed as both VV and VH. The gap is part of the benchmark and is not
  corrected in code.

## Results

Scores in % of the last-epoch model (there is no checkpoint selection), as mean ± sample standard
deviation over three seeds (`--seed 0`, `1` and `2`), on the val and test splits of the course data (1,426
and 2,065 images). Classification: accuracy and macro-F1. Detection: COCO mAP, AP@[.50:.95], and AP50,
AP@.50. The seeds differ by a few tenths of a point, so smaller differences are noise.

Δm is the leaderboard's ranking score: the mean relative gain in macro-F1 and mAP over a fixed reference, in
%. The leaderboard's reference is the **seed-0** run of ViT-B/16 with full fine-tuning (test: macro-F1 95.43,
mAP 31.67; val: 94.47 / 31.57). The Δm column is computed per seed against these values, then averaged; the
3-seed mean of the reference configuration is a little higher than its seed-0 run, so its Δm is slightly above 0.

The parameter column counts the trainable backbone parameters. Every run also trains the 20.1 M parameters of
the heads (neck, RPN, RoI heads, classifier), so `trainable_parameters` in `metrics.json` is the column value
+ 20.1 M.

Test:

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 95.9 ± 0.3 | 95.8 ± 0.3 | 31.8 ± 0.2 | 62.4 ± 0.1 | +0.3 ± 0.4 |
| vit | pretrained | lora | 0.88 | 94.2 ± 0.2 | 94.0 ± 0.2 | 25.4 ± 0.5 | 53.9 ± 0.4 | −10.6 ± 0.7 |
| vit | pretrained | moelora | 1.03 | 93.7 ± 0.5 | 93.4 ± 0.5 | 26.8 ± 0.3 | 55.9 ± 0.5 | −8.7 ± 0.7 |
| terramind | pretrained | full | 85.3 | 95.8 ± 0.2 | 95.6 ± 0.2 | 30.8 ± 0.4 | 61.8 ± 0.4 | −1.3 ± 0.6 |
| terramind | pretrained | lora | 0.88 | 93.1 ± 0.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | 55.4 ± 1.2 | −9.7 ± 0.9 |
| terramind | pretrained | moelora | 1.03 | 92.9 ± 0.1 | 92.4 ± 0.1 | 27.8 ± 0.3 | 57.7 ± 0.4 | −7.7 ± 0.5 |
| vit | scratch | full | 86.0 | 83.9 ± 0.6 | 82.8 ± 0.7 | 17.0 ± 0.3 | 36.7 ± 0.4 | −29.8 ± 0.8 |
| terramind | scratch | full | 85.3 | 86.7 ± 0.2 | 86.1 ± 0.2 | 17.8 ± 0.5 | 38.9 ± 0.4 | −26.9 ± 0.7 |

Val:

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 94.7 ± 0.3 | 94.4 ± 0.3 | 32.0 ± 0.4 | 62.7 ± 0.4 | +0.6 ± 0.6 |
| vit | pretrained | lora | 0.88 | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | −9.2 ± 0.8 |
| vit | pretrained | moelora | 1.03 | 92.6 ± 0.8 | 92.2 ± 0.8 | 27.1 ± 0.2 | 56.4 ± 0.3 | −8.2 ± 0.8 |
| terramind | pretrained | full | 85.3 | 94.5 ± 0.2 | 94.2 ± 0.3 | 31.5 ± 0.1 | 62.4 ± 0.0 | −0.3 ± 0.2 |
| terramind | pretrained | lora | 0.88 | 92.6 ± 0.4 | 92.2 ± 0.4 | 27.0 ± 0.5 | 55.9 ± 0.8 | −8.5 ± 1.1 |
| terramind | pretrained | moelora | 1.03 | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | −5.7 ± 0.3 |
| vit | scratch | full | 86.0 | 83.8 ± 0.8 | 83.5 ± 0.9 | 18.2 ± 0.5 | 39.0 ± 0.7 | −26.9 ± 1.1 |
| terramind | scratch | full | 85.3 | 86.0 ± 1.0 | 86.0 ± 0.9 | 19.2 ± 0.4 | 41.4 ± 0.4 | −24.2 ± 1.0 |
