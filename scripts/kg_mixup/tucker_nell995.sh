#!/bin/bash

cd ../..


python src/main.py --model tucker --dataset nell_995 --run-pretrain tucker_nell995_pretrain \
                   --lr 5e-5 --synth-weight 1e-2 --rel-dim 100 --input-drop 0.3 --hid-drop1 0.2 --hid-drop2 .2 \
                   --label-smooth 0 --threshold 25 --max-generate 5 --epochs 300 --validation 50 --save-every 50 --early-stop 10 \
                   --swa --swa-lr 1e-5 --save-as tucker_nell995_kg_mixup
