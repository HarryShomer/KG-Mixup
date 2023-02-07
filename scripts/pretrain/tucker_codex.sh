#!/bin/bash

cd ../..

python src/main.py --model tucker --dataset codex_m --strategy "none" \
                   --epochs 500 --train-type "1-N" --lr 1e-3 --decay 0.99 --rel-dim 100 \
                   --input-drop 0.5 --hid-drop1 0.02 --hid-drop2 0.4 --no-bias \
                   --early-stop 15 --save-as "tucker_codex_pretrain" 