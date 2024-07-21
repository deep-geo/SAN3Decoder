#!/usr/bin/env bash

python train.py \
--work_dir "/root/autodl-tmp/workdir" \
--run_name "SAN_3decoder" \
--seed 42 \
--epochs 500 \
--batch_size 16 \
--num_workers 16 \
--eval_interval 515 \
--test_sample_rate 0.3 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/autodl-tmp/ALL_Multi" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'recall'  'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/workdir/models/sup_CEP_baseline_No_MoNuSeg2020_07-01_19-33/epoch0079_step40500_test-loss0.1233_sam.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--edge_point_num 10 \
--iter_point 8 \
--encoder_adapter \
--multimask \
#--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \