#!/usr/bin/env bash

python train.py \
--work_dir "workdir" \
--run_name "nuclei" \
--seed 42 \
--epochs 100000 \
--batch_size 8 \
--num_workers 8 \
--log_interval 100 \
--test_sample_rate 1.0 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL_Multi" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--encoder_adapter \
--multimask \
--activate_unsupervised \
--unsupervised_dir "/root/autodl-tmp/datasets/SAM_nuclei/<unsupervised_root>" \
--unsupervised_start_epoch 0 \
--unsupervised_step 1 \
--unsupervised_weight 1.0 \
--unsupervised_weight_gr 0.0 \
--unsupervised_sample_rates 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--unsupervised_num_processes 2 \
--pred_iou_thresh 0.88 \
--stability_score_thresh 0.95 \
--points_per_side 32 \
--points_per_batch 256 \
#--unsupervised_only \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
