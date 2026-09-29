#!/usr/bin/env bash
# The eight reference runs, one after another. Extra arguments are passed to every run,
# e.g. a quick end-to-end check: bash run_all.sh --limit 64 --epochs 1 --out runs/smoke
set -e
# A pretrained backbone is fine-tuned gently (--backbone-lr 2e-5); new modules, adapters and scratch backbones use 1e-4.
python train.py --backbone vit --init pretrained --adapt full --backbone-lr 2e-5 "$@"
python train.py --backbone vit --init pretrained --adapt lora "$@"
python train.py --backbone vit --init pretrained --adapt moelora "$@"
python train.py --backbone terramind --init pretrained --adapt full --backbone-lr 2e-5 "$@"
python train.py --backbone terramind --init pretrained --adapt lora "$@"
python train.py --backbone terramind --init pretrained --adapt moelora "$@"
python train.py --backbone vit --init scratch --adapt full "$@"
python train.py --backbone terramind --init scratch --adapt full "$@"
