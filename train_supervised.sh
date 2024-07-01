#!/usr/bin/env bash

python train.py \
--work_dir "/root/tmp/workdir" \
--run_name "sup_CEP" \
--seed 42 \
--epochs 500 \
--batch_size 16 \
--num_workers 16 \
--eval_interval 500 \
--test_sample_rate 0.2 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/tmp/ALL_Multi" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'recall'  'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--edge_point_num 4
--iter_point 8 \
--encoder_adapter \
--multimask \
