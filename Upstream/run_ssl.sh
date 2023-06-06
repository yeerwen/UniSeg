#!/bin/bash

Traniner="UniSeg_Trainer"

GPU=0

task_exp=$Traniner

CUDA_VISIBLE_DEVICES=$GPU nnUNet_n_proc_DA=32 nnUNet_train 3d_fullres $Traniner 97 0 -exp_name $task_exp

python UniSeg_Metrics_test.py --result_path $task_exp'/3d_fullres/Task097_11task/'$Traniner'__DoDNetPlans/fold_0/validation_raw/'