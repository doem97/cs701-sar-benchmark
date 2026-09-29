# Details of the reference code

Back to the [README](../README.md).

## Model

```
image [B,1,512,512] -> backbone -> tokens [B,1024,768]   (32 x 32 patches of 16 px)
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens as a [B,768,32,32] map -> SimpleFeaturePyramid (strides 4 to 64, 256 channels)
                  -> RegionProposalNetwork + RoIHeads (torchvision Faster R-CNN, 9 classes + background)
loss = cross-entropy + the four Faster R-CNN losses
```

![The reference model](reference_model.png)

**Backbones** (`--backbone`), both ViT-B/16:

- `vit`: timm's `vit_base_patch16_224.augreg_in21k` (ImageNet-21k).
- `terramind`: TerraMind-1.0-base (IBM/ESA), an Earth-observation model, through its Sentinel-1 GRD input.

**Adaptation** (`--adapt`):

- `full`: every backbone weight trains.
- `lora`: the backbone is frozen; the `qkv` and `proj` layers of every attention block get
  `y = W x + (alpha / r) B A x`, with r = 16 and alpha = 32 (0.88 M trainable backbone parameters).
- `moelora`: as `lora`, but the rank-16 update is split into 4 experts of rank 4 (alpha = 8, so alpha / r = 2
  as in `lora`), mixed per token by a softmax router. Each task has its own router (1.03 M trainable backbone
  parameters).

Every run also trains the 20.1 M parameters of the heads (neck, RPN, RoI heads, classifier), so
`trainable_parameters` in `metrics.json` is the backbone count + 20.1 M.

## Training protocol (the same for all eight runs)

- AdamW, weight decay 0.05, batch size 16, bf16 autocast, 24 epochs.
- Learning rate 1e-4; the backbone of the two pretrained `full` runs uses 2e-5 (`--backbone-lr 2e-5`, as in
  `run_all.sh`), so a pretrained backbone is fine-tuned gently.
- One epoch of linear warmup, then cosine decay to 0, updated every iteration.
- Validation every 4 epochs and after the last one; test once, at the end. The reported model is the
  last-epoch model: there is no checkpoint selection.

## Design notes

- **Both labels on every image**, so every training batch trains both heads.
- **512 px input.** Every image is resized to 512 x 512 (non-square images are stretched): 1,024 tokens.
  Objects are small (median box 15.9 px at 512 px), so the smallest anchors are 8 px, on the stride-4 level.
  Predicted boxes are mapped back to original pixels before COCO scoring.
- **Normalization.** Images are read as one gray channel. Both backbones get the same mean and std
  (0.2526, 0.2133), computed over the 9,392 train images at 512 px.
- **Precision rule.** Backbone, neck and classification head run in bf16; the RPN and RoI heads run in fp32.
  torchvision's box coder casts anchors to the dtype of the regression output, and bf16 numbers between 256
  and 512 are 2 px apart: too coarse for objects of 6-15 px. Importing terratorch (`sarbench/backbones.py` does, for
  every run) switches fp32 matrix multiplications on the GPU to TF32; box coordinates are decoded
  elementwise in fp32 and are unaffected.
- **MoE-LoRA runs the backbone twice.** Its routers are task-specific, so the two tasks see different
  features and every forward pass runs the backbone once per task.
- **TerraMind domain gap.** TerraMind was pretrained on Sentinel-1 GRD backscatter in dB (polarizations VV
  and VH, 10 m pixels). Our images are 8-bit gray display images, mostly at a higher resolution; the gray
  channel is fed as both VV and VH. The gap is part of the benchmark and is not corrected in code.

## Time and memory

On one RTX PRO 6000 Blackwell GPU, a run alone takes about 50 min (LoRA about 47 min, MoE-LoRA about 1 h);
all eight in sequence take about 7 h. A run keeps about 11-17 GiB of GPU memory in use; TerraMind with
MoE-LoRA needs the most and does not fit on a 16 GB card. A smaller `--batch-size` saves memory but changes
the protocol.

## What a run writes

`runs/<backbone>_<init>_<adapt>/`:

- `log.txt`: training progress and the val and test scores.
- `metrics.json`: arguments, parameter counts, `train_hours` (wall time of the epoch loop, including the
  periodic validation; it grows if the GPU is shared), `peak_gpu_memory_gb` (`torch.cuda.max_memory_allocated`
  in GiB; the process needs about 2.5 GiB more), every validation result, and the final val and test metrics.
- `predictions_val.json`, `predictions_test.json`: detections in COCO result format, in original pixels.
- `model.pt`: the final weights (`state_dict`).

## Checks

```bash
bash run_all.sh --limit 64 --epochs 1 --out runs/smoke   # all eight configurations on 64 images per split, a few minutes
python -m pytest tests                                   # the tests (need a GPU)
```

`test_data.py`, `test_metrics.py` and `test_model.py` read the data in TA Zichen's layout (see the README),
so they need the same adaptation as the code; `test_backbones.py` and `test_adapters.py` need no data.

## Results in full

Scores in %, mean ± sample standard deviation over seeds 0, 1 and 2, last-epoch model. Accuracy and macro-F1
for classification; COCO mAP (AP@[.50:.95]) and AP50 for detection. The seeds differ by a few tenths of a
point, so smaller differences are noise. Δm is computed per seed against the reference, the mean of the three
ViT-B/16 + LoRA runs (test 93.99 / 25.41, val 93.31 / 26.13; the Codabench leaderboard uses the same values), then
averaged, so the reference configuration shows 0.

Test (2,065 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 95.9 ± 0.3 | 95.8 ± 0.3 | 31.8 ± 0.2 | 62.4 ± 0.1 | +13.5 ± 0.4 |
| vit | pretrained | lora | 0.88 | 94.2 ± 0.2 | 94.0 ± 0.2 | 25.4 ± 0.5 | 53.9 ± 0.4 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 93.7 ± 0.5 | 93.4 ± 0.5 | 26.8 ± 0.3 | 55.9 ± 0.5 | +2.5 ± 0.8 |
| terramind | pretrained | full | 85.3 | 95.8 ± 0.2 | 95.6 ± 0.2 | 30.8 ± 0.4 | 61.8 ± 0.4 | +11.5 ± 0.7 |
| terramind | pretrained | lora | 0.88 | 93.1 ± 0.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | 55.4 ± 1.2 | +1.4 ± 1.0 |
| terramind | pretrained | moelora | 1.03 | 92.9 ± 0.1 | 92.4 ± 0.1 | 27.8 ± 0.3 | 57.7 ± 0.4 | +3.8 ± 0.7 |
| vit | scratch | full | 86.0 | 83.9 ± 0.6 | 82.8 ± 0.7 | 17.0 ± 0.3 | 36.7 ± 0.4 | −22.5 ± 0.9 |
| terramind | scratch | full | 85.3 | 86.7 ± 0.2 | 86.1 ± 0.2 | 17.8 ± 0.5 | 38.9 ± 0.4 | −19.3 ± 0.8 |

Val (1,426 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 94.7 ± 0.3 | 94.4 ± 0.3 | 32.0 ± 0.4 | 62.7 ± 0.4 | +11.8 ± 0.7 |
| vit | pretrained | lora | 0.88 | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 92.6 ± 0.8 | 92.2 ± 0.8 | 27.1 ± 0.2 | 56.4 ± 0.3 | +1.3 ± 0.9 |
| terramind | pretrained | full | 85.3 | 94.5 ± 0.2 | 94.2 ± 0.3 | 31.5 ± 0.1 | 62.4 ± 0.0 | +10.7 ± 0.3 |
| terramind | pretrained | lora | 0.88 | 92.6 ± 0.4 | 92.2 ± 0.4 | 27.0 ± 0.5 | 55.9 ± 0.8 | +1.0 ± 1.3 |
| terramind | pretrained | moelora | 1.03 | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | +4.4 ± 0.3 |
| vit | scratch | full | 86.0 | 83.8 ± 0.8 | 83.5 ± 0.9 | 18.2 ± 0.5 | 39.0 ± 0.7 | −20.3 ± 1.3 |
| terramind | scratch | full | 85.3 | 86.0 ± 1.0 | 86.0 ± 0.9 | 19.2 ± 0.4 | 41.4 ± 0.4 | −17.3 ± 1.2 |
