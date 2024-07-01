#!/usr/bin/env bash

python train.py \
--work_dir "/root/tmp/workdir" \
--run_name "Unsupervised_500K_dynamic" \
--seed 42 \
--epochs 100 \
--batch_size 16 \
--num_workers 16 \
--log_interval 100 \
--test_sample_rate 0.2 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/tmp/ALL" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "/root/tmp/workdir/models/sup_clust_edge_enc_06-08_10-57/epoch0079_test-loss0.1429_sam.pth" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--encoder_adapter \
--multimask \
--activate_unsupervised \
--unsupervised_dir "/root/tmp/CEM500K/Crop256/" \
--unsupervised_start_epoch 1 \
--unsupervised_step 1 \
--unsupervised_weight 1.0 \
--unsupervised_weight_gr 0.0 \
--unsupervised_initial_sample_rate 0.1 \
--unsupervised_sample_rate_delta 0.05 \
--unsupervised_metric_delta_threshold 0.01 \
--unsupervised_num_processes 2 \
--unsupervised_focused_metric "Overall/aji" \
--pred_iou_thresh 0.88 \
--stability_score_thresh 0.95 \
--points_per_side 32 \
--points_per_batch 256 \
#--unsupervised_only \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
