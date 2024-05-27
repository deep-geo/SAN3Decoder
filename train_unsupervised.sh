#!/usr/bin/env bash

python train.py \
--work_dir "workdir" \
--run_name "nuclei" \
--seed 42 \
--epochs 100000 \
--batch_size 8 \
--num_workers 8 \
--image_size 256 \
--mask_num 5 \
--split_paths "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL/split.json" \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' \
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
--unsupervised_pred_iou_thresh 0.88 \
--unsupervised_stability_score_thresh 0.95 \
--unsupervised_weight_gr 0.1 \
#--unsupervised_only \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
