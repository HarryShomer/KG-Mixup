#!/bin/bash

cd ../..


python src/main.py --model tucker --dataset fb15k_237 --run-pretrain tucker_fb15k237_pretrain \
                   --threshold 5 --synth-weight 1 --lr 5e-5 --decay 0.99  --epochs 300 --validation 50 --input-drop 0.3 \
                   --save-as tucker_fb15k237_kg_mixup --epochs 300 --validation 50 --save-every 100 --early-stop 10 \
                   --max-generate 5 --swa --swa-lr 5e-4