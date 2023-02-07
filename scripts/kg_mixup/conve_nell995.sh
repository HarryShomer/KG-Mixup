#!/bin/bash

cd ../..

python src/main.py --dataset nell_995 --run-pretrain conve_nell995_pretrain --lr 5e-5  \
                   --synth-weight 1 --input-drop 0.4 --feat-drop 0.3 --hid-drop 0.1 \
                   --label-smooth 0.1 --threshold 5 --max-generate 5 --swa --swa-lr 1e-5 \
                   --epochs 300 --validation 50 --save-every 50 --early-stop 10 --save-as conve_nell995_kg_mixup
