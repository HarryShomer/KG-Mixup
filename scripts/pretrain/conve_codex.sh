#!/bin/bash

cd ../..

python src/main.py --model conve --dataset codex_m --strategy "none" \
                   --epochs 500 --train-type "1-K" --lr 5e-4 --decay 0.99 --label-smooth 0.1 \
                   --input-drop 0.2 --hid-drop 0.0 --feat-drop 0.4 \
                   --early-stop 15 --save-as "conve_codex_pretrain" 