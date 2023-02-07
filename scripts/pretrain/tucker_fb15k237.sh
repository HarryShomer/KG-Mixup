#!/bin/bash

cd ../..

python src/main.py --model tucker --dataset fb15k_237 --strategy "none" \
                   --epochs 500 --train-type "1-N" --lr 5e-4 --label-smooth 0.1 \
                   --input-drop 0.3 --hid-drop1 0.4 --hid-drop2 0.5 \
                   --early-stop 15 --save-as "tucker_fb15k237_pretrain" 