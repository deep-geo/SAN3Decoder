#!/usr/bin/env bash

python inference.py \
--random_seed 42 \
--sam_checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--checkpoint "" \
--model_type "vit_b" \
--device "cuda" \
--image_size 256 \
--data_root "/root/autodl-tmp/zero_shot/MoNuSeg2020/" \
--batch_size 8 \
--num_workers 8 \
--metrics "dice" "iou" "f1_score" "precision" "recall" "accuracy" "aji" "dq" "sq" "pq" \
--point_num 1 \
--boxes_prompt \
--iter_point 8 \
--multimask \
--pred_iou_thresh 0.88 \
--stability_score_thresh 0.95 \
--points_per_side 32 \
--points_per_batch 256

#--checkpoint "/root/autodl-tmp/workdir/models/sup_clust_edge_enc_06-02_20-42/epoch0065_test-loss0.1528_sam.pth" \
# autodl-tmp/
#autodl-tmp/sam_vit_b_01ec64.pth sam-med2d_b.pth