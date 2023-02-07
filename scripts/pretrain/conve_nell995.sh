#!/bin/bash

cd ../..

python src/main.py --model conve --dataset nell_995 --strategy "none" \
                   --epochs 500 --train-type "1-K" --lr 5e-4 --decay 0.99 \
                   --input-drop 0.15 --hid-drop 0.35 --feat-drop 0.5 \
                   --early-stop 15 --save-as "conve_nell995_pretrain" 