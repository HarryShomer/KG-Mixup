#!/bin/bash

cd ../..

python src/main.py --model conve --dataset fb15k_237 --strategy "none"  \
                   --epochs 500 --train-type "1-N" --lr 1e-3 --label-smooth 0.1 \
                   --early-stop 15 --save-as "conve_fb15k237_pretrain" 