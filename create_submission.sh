#!/usr/bin/env bash

# Run experiments: the submission for each exp will be created in "/submissions"
CUDA_VISIBLE_DEVICES=1 python naive_baseline.py --scenario="ni" --sub_dir="ni" -cls='mnasnet' --optimizer='adam' --lr=1e-4 --aug=True --epochs=2 #--regularize_mode='EWC' --icarl=True
CUDA_VISIBLE_DEVICES=1 python naive_baseline.py --scenario="multi-task-nc" --sub_dir="multi-task-nc" -cls='mnasnet' --replay_examples=1000 --aug=True --regularize_mode='EWC' --icarl=True
CUDA_VISIBLE_DEVICES=1 python naive_baseline.py --scenario="nic" --sub_dir="nic" --epochs=1 --replay_examples=20 --aug=True

# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && zip -r ../submission.zip ./ni ./multi-task-nc ./nic

