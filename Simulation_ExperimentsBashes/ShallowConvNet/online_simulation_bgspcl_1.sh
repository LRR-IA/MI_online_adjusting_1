#!/bin/bash

program_path="Online_simulation/Online_bgspcl.py"
workspace_folder="/home/jyt/workspace/MI_Online_Adjusting_1"
cd $workspace_folder

# 超参数列表
T_distil_list=(2.0)  # different temperature values for distillation
tau_cons_list=(1.0)  # different consistency loss weights

for T_distil in "${T_distil_list[@]}"
do
  for tau_cons in "${tau_cons_list[@]}"
  do
    result_subfolder="T_${T_distil}tau${tau_cons}"
    Online_result_save_rootdir="Online_simulation_experiments/ShallowConvNet/ShallowConvNet_bgspcl_1"
    for i in $(seq 1 25)
    do
      sub_name=$(printf "%03d" $i)  
      python3 $program_path \
        --seed 3407 \
        --gpu_idx 0 \
        --sub_name $sub_name \
        --Offline_folder_path "/data/datasets_Jyt/datasets_MI/hand_elbow/derivatives/" \
        --windows_num 120 \
        --trial_pre 120 \
        --proportion 0.75 \
        --Offline_result_save_rootdir "Offline_simulation_experiments/ShallowConvNet/ShallowConvNet_basemodel" \
        --Online_folder_path "Online_DataCollected" \
        --Online_result_save_rootdir "$Online_result_save_rootdir" \
        --restore_file "pretrained_weights/checkpoints_test_predict/checkpoints_test_encoder3_light/encoder_epoch_1.0.pt" \
        --n_epoch_offline 100 \
        --n_epoch_online 2 \
        --batch_size 16 \
        --mode "online" \
        --batch_size_online 9 \
        --trial_nums 96 \
        --best_validation_path "lr0.01_dropout0.5" \
        --unfreeze_encoder_offline "True" \
        --unfreeze_encoder_online "True" \
        --alpha_distill 0.50 \
        --update_trial 1 \
        --update_wholeModel 12 \
        --para_m 0.99 \
        --cons_rate 1.0 \
        --preprocess_norm "True" \
        --T_distil $T_distil \
        --tau_cons $tau_cons \
        --model_type "ShallowConvNet" \
        --queue_size 36 \
        --bg_init_lambda 0.125 \
        --bg_spcl_lr 0.05 \
        --bg_spcl_round 19 \
        --bg_k 3.0 \
        --bg_sampling_rate 256 \
        --bg_trial_len 4 \
        --bg_update_trial 12
    done
  done
done