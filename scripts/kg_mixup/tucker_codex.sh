#!/bin/bash

cd ../..


python src/main.py --model tucker --dataset codex_m --run-pretrain tucker_codex_pretrain \
                   --lr 1e-5 --decay 0.995 --label-smooth 0 --synth-weight 1 --rel-dim 100 --no-bias \
                   --input-drop .3 --hid-drop1 0.5 --hid-drop2 .5 --threshold 5 --max-generate 5 \
                   --epochs 250 --validation 50 --save-every 50 --early-stop 10  --swa --swa-lr 5e-4 --save-as tucker_codex_kg_mixup 
