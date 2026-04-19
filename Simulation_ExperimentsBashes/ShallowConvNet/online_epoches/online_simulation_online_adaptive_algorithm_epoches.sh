#!/bin/bash

program_path="Online_simulation/Online_adaptive_algorithm_1.py"
workspace_folder="/home/jyt/workspace/MI_Online_Adjusting_1"
cd $workspace_folder

# 超参数列表
T_distil_list=(2 4)  # different temperature values for distillation
tau_cons_list=(4 6 8)  # different consistency loss weights

for T_distil in "${T_distil_list[@]}"
do
  for tau_cons in "${tau_cons_list[@]}"
  do
    result_subfolder="ntrial_${T_distil}_nsession${tau_cons}"
    Online_result_save_rootdir="Online_simulation_experiments/ShallowConvNet/epoches_sensitivity/ShallowConvNet_Online_adaptive_algorithm/${result_subfolder}"
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
        --n_epoch_online $T_distil \
        --n_epoch_online_1 $tau_cons \
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
        --para_m 0.90 \
        --cons_rate 1.0 \
        --preprocess_norm "True" \
        --T_distil 2.0 \
        --tau_cons 1.0 \
        --model_type "ShallowConvNet"
    done
  done
done