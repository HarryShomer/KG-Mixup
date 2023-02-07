#!/bin/bash

cd ../..

python src/main.py --dataset codex_m --run-pretrain conve_codex_pretrain --lr 1e-5  \
                   --synth-weight 1 --input-drop 0.1 --feat-drop 0.2 --hid-drop 0.3 \
                   --label-smooth 0.1 --threshold 5 --max-generate 5 --swa --swa-lr 1e-5 \
                   --epochs 250 --validation 50 --save-every 50 --early-stop 10 --save-as conve_codex_kg_mixup
