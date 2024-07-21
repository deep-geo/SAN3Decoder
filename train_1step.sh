#!/usr/bin/env bash

python train_step1.py \
--work_dir "/root/autodl-tmp/workdir" \
--run_name "SAN_3Decoder" \
--seed 42 \
--epochs 100 \
--batch_size 8 \
--test_sample_rate 0.1 \
--test_size 0.1 \
--num_workers 8 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/autodl-tmp/ALL_Multi/" \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "/root/autodl-tmp/workdir/models/sup_CEP_baseline_No_MoNuSeg2020_07-01_19-33/epoch0079_step40500_test-loss0.1233_sam.pth" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--edge_point_num 3 \
--iter_point 8 \
--encoder_adapter \
--multimask \
