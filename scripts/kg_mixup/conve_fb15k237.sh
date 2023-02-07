#!/bin/bash

cd ../..

python src/main.py --dataset fb15k_237 --run-pretrain conve_fb15k237_pretrain --lr 1e-4  \
                   --save-as conve_fb15k237_kg_mixup --epochs 400 --validation 50 --save-every 50 --early-stop 10 \
                   --threshold 5 --max-generate 5 --synth-weight 1 --swa --swa-lr 5e-4 --swa-every 10
                
