#!/usr/bin/env bash

dst_root="<dst_root>"
dst_size=256

# CoNIC
src_root="<path_to_CoNIC_src_root>"
dst_prefix="CoNIC"
echo -e "\n################# PROCESS CoNIC #################"
python preprocess_CoNIC.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# fluorescence
src_root="<path_to_fluorescence_src_root>"
dst_prefix="fluorescence"
echo -e "\n################# PROCESS fluorescence #################"
python preprocess_fluorescence.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# histology
src_root="<path_to_histology_src_root>"
dst_prefix="histology"
echo -e "\n################# PROCESS histology #################"
python preprocess_histology.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# thyroid
src_root="<path_to_thyroid_src_root>"
dst_prefix="thyroid"
echo -e "\n################# PROCESS thyroid #################"
python preprocess_thyroid.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# GlandSeg
src_root="<path_to_GlandSeg_src_root>"
dst_prefix="GlandSeg"
echo -e "\n################# PROCESS GlandSeg #################"
python preprocess_GlandSeg.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# DynamicNuclearNet
src_root="<path_to_DynamicNuclearNet_src_root>"
dst_prefix="DynamicNuclearNet"
echo -e "\n################# PROCESS DynamicNuclearNet #################"
python preprocess_DynamicNuclearNet.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix


# split dataset
echo -e "\n>>>>>>>>>>>>>>>>>>>>> Split Datasets <<<<<<<<<<<<<<<<<<<<<<"
python split_dataset.py --data_root $dst_root --ext "png" --test_size 0.05 --seed 42