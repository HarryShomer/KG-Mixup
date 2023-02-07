#!/bin/bash

cd ../..

python src/main.py --model tucker --dataset nell_995 --strategy "none" \
                   --epochs 500 --train-type "1-N" --lr 1e-3 --decay 0.99 --rel-dim 100 \
                   --label-smooth 0.1 --input-drop 0.05 --hid-drop1 0.2 --hid-drop2 0.2 \
                   --early-stop 15 --save-as "tucker_nell995_pretrain" 