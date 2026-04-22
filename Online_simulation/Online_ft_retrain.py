import os
import sys
import numpy as np
import torch
import torch.nn as nn

import time
import datetime
import argparse
import re
import shutil
import copy
from collections import deque

from easydict import EasyDict as edict
import torch.optim
from tqdm import trange
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import EEGNetFea, DeepConvNetFea, ShallowConvNetFea
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv, Online_simulation_read_csv, Online_simulation_read_csv_windows_preprocess_normalization
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, \
    save_pickle, train_one_epoch, train_one_epoch_fea, train_update, eval_model, eval_model_fea, train_one_epoch_MMD, save_training_curves_FixedTrainValSplit, \
        write_performance_info_FixedTrainValSplit, write_program_time, eval_model_confusion_matrix_fea, train_one_epoch_MMDavg, write_inference_time
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix, \
    eval_model_confusion_matrix, accuracy_iteration_plot,str2bool,accuracy_save2csv, train_one_epoch_MMD_Weights, compute_total_accuracy_per_class,\
    accuracy_perclass_save2csv, accuracy_perclass_iteration_plot, eval_model_fea_exemplars, eval_model_fea_exemplars_distillation, train_one_epoch_fea_distillation, train_one_epoch_fealogitlabel_distillation, \
        eval_model_fea_exemplars_distillation_label, eval_model_fea_exemplars_distillation_datafea_logitlabel, train_one_epoch_logit_distillation, train_one_epoch_label_distillation, train_one_epoch_logitlabel_distillation, train_one_epoch_fea_MMDContrastive, train_one_epoch_fea_MMDContrastive_targetcls_iter, \
            MultiClassFocalLoss, MultiClassNpFocalLoss, PolyLoss, write_exemplar_time, eval_model_fea_classPrototypes, plot_calibration_histogram
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch_confusionMatrix
from Online_simulation_synthesizing.Online_simulation_synthesizing_subjects import Online_simulation_synthesizing_results, Online_simulation_synthesizing_results_comparison,\
      Online_simulation_synthesizing_results_linear, Online_simulation_synthesizing_results_comparison_linear, Online_simulation_synthesizing_results_linear_perclass, Online_simulation_synthesizing_results_2cls_linear,\
      Online_simulation_synthesizing_results_comparison_linear_2cls, Online_simulation_synthesizing_results_polynomial, Online_simulation_synthesizing_results_comparison_polynomial, Online_simulation_synthesizing_results_comparison_polynomial_optimized, Online_simulation_synthesizing_results_polynomial_avg, Online_simulation_synthesizing_results_polynomial_avgF1,Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass, Online_simulation_synthesizing_results_calibration_avg, Online_simulation_synthesizing_results_calibration_perclass, Online_simulation_synthesizing_results_polynomial_avgF1_noRest, Online_simulation_synthesizing_results_polynomial_avgF1_Rest


#for personal model, save the test prediction of each cv fold
def Offline_EEGNet_simulation(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name = args_dict.sub_name
    Offline_folder_path = args_dict.Offline_folder_path
    trial_pre = args_dict.trial_pre
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    preprocess_norm = args_dict.preprocess_norm
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_offline = args_dict.n_epoch_offline
    batch_size = args_dict.batch_size
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    patience = args_dict.patience
    
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
    
    sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array, \
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=40, preprocess=preprocess_norm, proportion=proportion)
    
    sub_train_feature_array = sub_train_feature_array.astype(np.float32)
    sub_val_feature_array = sub_val_feature_array.astype(np.float32)

    #dataset object
    group_train_set = brain_dataset(sub_train_feature_array, sub_train_label_array)
    group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)

    #dataloader object
    cv_train_batch_size = batch_size
    cv_val_batch_size = batch_size
    sub_cv_train_loader = torch.utils.data.DataLoader(group_train_set, batch_size=cv_train_batch_size, shuffle=True) 
    sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False)
    print("data prepared")

    """
    The offline simulation part, using the data from MI2 as the simulation dataset 
    MI2: https://doi.org/10.7910/DVN/RBN3XG
    """
    #cross validation:
    lrs = [0.001, 0.01, 0.1]
    dropouts = [0.0, 0.25, 0.5, 0.75]

    start_time = time.time()
    
    for lr in lrs:
        for dropout in dropouts:
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
            print(experiment_name)
            #derived arg
            result_save_subjectdir = os.path.join(Offline_result_save_rootdir, sub_name, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
            result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
            result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
            result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)
            makedir_if_not_exist(result_save_subject_predictionsdir)
            makedir_if_not_exist(result_save_subject_resultanalysisdir)
            makedir_if_not_exist(result_save_subject_trainingcurvedir)
            
            result_save_dict = dict()
            
            #create model
            if preprocess_norm:
                input_feature_size = 30
            else:
                input_feature_size = 29
                
            if args_dict.model_type in ['EEGNetFea', 'EEGNet']:
                model = EEGNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
            elif args_dict.model_type in ['DeepConvNetFea', 'DeepConvNet']:
                model = DeepConvNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, dropout=dropout)
            elif args_dict.model_type in ['ShallowConvNetFea', 'ShallowConvNet']:
                model = ShallowConvNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, dropout=dropout)
                
            # reload weights from restore_file is specified
            if restore_file != 'None':
                #restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
                restore_path = restore_file
                print('loading checkpoint: {}'.format(restore_path))
            
            model = model.to(device)

            #create criterion and optimizer
            criterion = nn.CrossEntropyLoss()
            #criterion = MultiClassFocalLoss(device=device, alpha=[0.33,0.33,0.33])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
            #training loop
            best_val_accuracy = 0.0
            is_best = False
            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_accuracy = []
            
            epochs_no_improve = 0

            for epoch in trange(n_epoch_offline, desc='1-fold cross validation'):
                average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, sub_cv_train_loader, device)
                val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_cv_val_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, sub_cv_train_loader, device)

                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag based on overall validation accuracy
                is_best = val_accuracy > best_val_accuracy

                if is_best:
                    best_val_accuracy = val_accuracy
                    epochs_no_improve = 0

                    torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                    print('Model saved at epoch {} with validation accuracy: {:.2f}%'.format(epoch + 1, val_accuracy))

                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                    for cls_i in range(accuracy_per_class.shape[0]):
                        result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs. No improvement for {patience} epochs.')
                    break

            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #save the model at last epoch
            #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            
            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
    
    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(os.path.join(Offline_result_save_rootdir, sub_name), total_time)


def Online_updating_EEGNet_simulation(args_dict):
    """
    Online simulation part
    The online simulation part using the new weighting method to update the model
    """  
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name = args_dict.sub_name
    Offline_folder_path = args_dict.Offline_folder_path
    trial_pre = args_dict.trial_pre
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    preprocess_norm = args_dict.preprocess_norm
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_online = args_dict.n_epoch_online
    batch_size = args_dict.batch_size
    batch_size_online = args_dict.batch_size_online    # batch_size_online = 4
    trial_nums = args_dict.trial_nums    # trial_nums = 40
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    accuracy_per_class_init = args_dict.accuracy_per_class_init
    update_trial = args_dict.update_trial
    alpha_distill = args_dict.alpha_distill
    update_wholeModel = args_dict.update_wholeModel
    para_m = args_dict.para_m
    cons_rate = args_dict.cons_rate
    T_distil = args_dict.T_distil
    tau_cons = args_dict.tau_cons
    queue_size = args_dict.queue_size

    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
    
    sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array, \
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=50, preprocess=preprocess_norm,\
                                                                                                proportion=proportion, batch_size_online=batch_size_online, \
                                                                                                    pattern=  [1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 
                                                                                                               2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 
                                                                                                               1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 
                                                                                                               2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0])

    sub_train_feature_array = sub_train_feature_array.astype(np.float32)
    sub_val_feature_array = sub_val_feature_array.astype(np.float32)
    sub_train_feature_array_1 = sub_train_feature_array_1.astype(np.float32)

    match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", restore_file)
    if match:
        lr = float(match.group(1))
        dropout = float(match.group(2))
        print(f"lr={lr}, dropout={dropout}")
    else:
        print("No match found.")
    
    #create model
    if preprocess_norm:
        input_feature_size = 30
    else:
        input_feature_size = 29
    
    if args_dict.model_type in ['EEGNetFea', 'EEGNet']:
        model = EEGNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
    elif args_dict.model_type in ['DeepConvNetFea', 'DeepConvNet']:
        model = DeepConvNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, dropout=dropout)
    elif args_dict.model_type in ['ShallowConvNetFea', 'ShallowConvNet']:
        model = ShallowConvNetFea(feature_size=input_feature_size, num_timesteps=512, num_classes=3, dropout=dropout)

    #reload weights from restore_file is specified
    if restore_file != 'None':
        # move the best model from the offline experiments results
        Offline_path_encoder = os.path.join(Offline_result_save_rootdir, sub_name, restore_file, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations
        
        makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name, restore_file, 'checkpoint'))
        restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name, restore_file, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations

        # Copy Offline path to restore path
        shutil.copy(Offline_path_encoder, restore_path_encoder)
        print('Successfully copied {} to {}'.format(Offline_path_encoder, restore_path_encoder))

        # load the model
        model.load_state_dict(torch.load(restore_path_encoder))  
        print('loading checkpoint encoder: {}'.format(restore_path_encoder))
    
    # online simulation
    _combined_feature_array = np.concatenate((sub_train_feature_array, sub_val_feature_array), axis=0)
    _combined_label_array = np.concatenate((sub_train_label_array, sub_val_label_array), axis=0)
    # to avoid the situation that data out of offline training are used as the online training set earlier, we get the trial_pre num of the dataset
    unique_labels = np.unique(_combined_label_array)
    combined_feature_array = []
    combined_label_array = []
    for label in unique_labels:
        indices = np.where(_combined_label_array == label)[0]
        selected_indices = indices[:40*3]
        sub_feature = _combined_feature_array[selected_indices]
        sub_label = _combined_label_array[selected_indices]
        combined_feature_array.append(sub_feature)
        combined_label_array.append(sub_label)
    
    combined_feature_array = np.concatenate(combined_feature_array, axis=0)
    combined_label_array = np.concatenate(combined_label_array, axis=0)

    experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
    print(experiment_name)
    #derived arg
    result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
    result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
    result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
    result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
    result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

    makedir_if_not_exist(result_save_subjectdir)
    makedir_if_not_exist(result_save_subject_checkpointdir)
    makedir_if_not_exist(result_save_subject_predictionsdir)
    makedir_if_not_exist(result_save_subject_resultanalysisdir)
    makedir_if_not_exist(result_save_subject_trainingcurvedir)

    predict_accuracies = []
    class_predictions_arrays = []
    labels_arrays = []
    probabilities_arrays = np.empty((0, 3))

    # accuracies_per_class = []
    accuracy_per_class_iters = []

    _n_epoch_online = n_epoch_online
    accuracies_per_class_iterations = []
    accuracies_per_class_iterations.append([0, 0])
    accuracies_per_class_iterations.append([1, 0])
    accuracies_per_class_iterations.append([2, 0])
    accuracies_per_class_iterations_Rest = []
    accuracies_per_class_iterations_Rest.append([0, accuracy_per_class_init[0]])  # saving the existing initial caliberation accuracy for class 0
    

    # this is the implementation of paper
    # Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. Nature communications, 16(1), 5401. https://doi.org/10.1038/s41467-025-61064-x
    # =========================================================================
    # NC论文复现：在线finetune + retrain 配置
    # 结构说明：
    #   每个session = 12个trial，每个trial = 9个segments（batch_size_online=9）
    # 触发逻辑：
    #   - session内前6个trial完成后 → mid-session finetune
    #       仅用当前session前6个trial的在线数据，冻结firstConv+depthwiseConv
    #       lr=0.0001，epoch≤100，早停patience=80
    #   - 每个完整session（12个trial）结束后 → full retrain
    #       用离线数据全部 + 所有历史在线trial数据，全参数更新
    #       lr=0.001，epoch≤300，早停patience=80
    #   - 训练/验证集均按trial级别8:2随机划分（绝不按segment划分，防数据泄露）
    # =========================================================================
    TRIALS_PER_SESSION = 12      # 每个session包含的trial数
    FINETUNE_TRIGGER   = 6       # session内触发finetune的trial数（前半session）
    FINETUNE_LR        = 0.0001  # finetune学习率（原论文低10倍）
    FINETUNE_DROPOUT   = 0.65    # finetune dropout（防小数据集过拟合，原论文设定）
    FINETUNE_EPOCHS    = 64      # finetune最大epoch上限
    FINETUNE_PATIENCE  = 32      # finetune早停patience
    RETRAIN_LR         = 0.001   # retrain学习率
    RETRAIN_DROPOUT    = 0.5     # retrain dropout（原论文 Orig 模式固定为0.5）
    RETRAIN_EPOCHS     = 100     # retrain最大epoch上限
    RETRAIN_PATIENCE   = 50     # retrain早停patience

    # EEGNetFea 的 named_children() 共4个顶层模块：
    #   firstConv     → Conv2d(时域卷积) + BN2d         → finetune时冻结
    #   depthwiseConv → DepthwiseConv + BN2d + ELU + Pool + Dropout → finetune时冻结
    #   separableConv → SeparableConv + BN2d + ELU + Pool + Dropout → finetune时可更新
    #   classifier    → Flatten + Linear                → finetune时可更新
    if args_dict.model_type in ['EEGNetFea', 'EEGNet']:
        # 冻结时域卷积(firstConv: Conv2d+BN) + 空域卷积(depthwiseConv: DepthwiseConv+BN+ELU+Pool+Dropout)
        # 对应原论文 Keras EEGNet 冻结前4个有参数层（Conv2D+BN+DepthwiseConv2D+BN）
        # ELU/Pool/Dropout 无参数，按模块名冻结与原论文等价
        FREEZE_MODULES = {'firstConv', 'depthwiseConv'}
    elif args_dict.model_type in ['DeepConvNetFea', 'DeepConvNet']:
        # DeepConvNetFea: block1 = Conv2d(时域) + Conv2d(空域) + BN + ELU + MaxPool
        # block1 对应 EEGNet 的 firstConv+depthwiseConv，是跨session最稳定的低层特征
        # block2/block3/block4/classifier 对应高层融合层+分类头，finetune时可更新
        FREEZE_MODULES = {'block1'}
    elif args_dict.model_type in ['ShallowConvNetFea', 'ShallowConvNet']:
        # ShallowConvNetFea: block1 = Conv2d(时域) + Conv2d(空域) + BN
        # 整体极浅（仅block1+pool+drop+classifier），block1是唯一的低层特征提取模块
        # pool/drop无参数，classifier对漂移敏感，finetune时只更新classifier
        FREEZE_MODULES = {'block1'}

    # 记录初始离线数据的segment数，用于retrain时区分离线/在线部分
    # combined_feature_array在函数入口处已初始化为离线数据
    n_offline_segs_init = combined_feature_array.shape[0]

    # base_model_state_dict：每个session开始时的Base Model权重
    # 前半段(trial 1-6)用Base Model推理，后半段(trial 7-12)用Fine-tuned Model推理
    # 第一个session的Base Model = 离线训练好的权重（已在函数入口加载）
    base_model_state_dict = copy.deepcopy(model.state_dict())
    # finetuned_state_dict：每个session前半段结束后finetune得到的权重
    finetuned_state_dict = None

    for trial_idx in range(trial_nums):
        # generate the new data, simulating the online experiment 
        sub_train_feature_batches = sub_train_feature_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online, :, :]
        sub_train_label_batches = sub_train_label_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online]

        # 将当前trial的在线数据累积进combined数组
        # 累积后：combined_feature_array = 离线数据(初始) + 所有历史在线trial数据(segment级别)
        combined_feature_array = np.concatenate((combined_feature_array, sub_train_feature_batches), axis=0)
        combined_label_array   = np.concatenate((combined_label_array,   sub_train_label_batches),   axis=0)
        unique_labels = np.unique(combined_label_array)

        # 计算当前trial在session内的局部位置（1-based，范围1~TRIALS_PER_SESSION）
        # 必须在推理前计算，以决定使用哪个模型权重
        trial_in_session = (trial_idx % TRIALS_PER_SESSION) + 1

        # -----------------------------------------------------------------------
        # 对应原论文在线逻辑：
        #   前半段 (trial_in_session 1~6)  → 使用当前session的 Base Model 推理
        #   后半段 (trial_in_session 7~12) → 使用 Fine-tuned Model 推理
        #   session开始(trial_in_session=1)时，加载该session的Base Model权重
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        # 对应原论文在线推理逻辑：
        #   session开始(trial_in_session=1)时切换到新的Base Model
        #   前半段 (trial_in_session 1~6)  → Base Model 推理（不重复load，session开始时load一次）
        #   后半段 (trial_in_session 7~12) → Fine-tuned Model 推理（finetune触发后即切换）
        # -----------------------------------------------------------------------
        if trial_in_session == 1:
            # 新session开始：切换到该session的Base Model（retrain结束后已更新）
            model.load_state_dict(copy.deepcopy(base_model_state_dict))
            model = model.to(device)
            print('[Session {}] New session started, loaded Base Model weights.'.format(
                trial_idx // TRIALS_PER_SESSION + 1))
        elif trial_in_session == FINETUNE_TRIGGER + 1:
            # 后半段开始（trial_in_session=7）：切换到Fine-tuned Model
            # finetune已在trial_in_session=6推理结束后完成并保存
            if finetuned_state_dict is not None:
                model.load_state_dict(copy.deepcopy(finetuned_state_dict))
                model = model.to(device)
                print('[Session {}] Switching to Fine-tuned Model at trial_in_session={}.'.format(
                    trial_idx // TRIALS_PER_SESSION + 1, trial_in_session))
            else:
                # 理论上不应发生（finetune在trial_in_session=6时一定已触发）
                print('[Warning] finetuned_state_dict is None at trial_in_session=7, using Base Model.')
        # 其余trial（2~6前半段，8~12后半段）模型权重不变，沿用上一trial的模型

        # online simulation trials
        print("********** Online simulation trial: {} ***********".format(trial_idx))
        start_time_infer = time.time()
        model = model.to(device)
        model.eval()
        # online testing of the MI class
        ground_truth_label = np.unique(sub_train_label_batches)
        print("ground truth label:{}".format(ground_truth_label))
        
        # form the online test set 
        _sub_updating_predict = brain_dataset(sub_train_feature_batches, sub_train_label_batches)
        sub_updating_predict = torch.utils.data.DataLoader(_sub_updating_predict, batch_size=sub_train_feature_batches.shape[0], shuffle=False)

        predict_accu, class_predictions_array, labels_array, probabilities_array, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_updating_predict, device)
        # recording the corresponding accuracy of each class
        accuracies_per_class_iterations.append([ground_truth_label[0], predict_accu/100])
        predict_accuracies.append(predict_accu)
        probabilities_arrays = np.vstack((probabilities_arrays, probabilities_array))
        class_predictions_arrays.extend(class_predictions_array.tolist())
        labels_arrays.extend(labels_array.tolist())
        #accuracies_per_class.append(accuracy_per_class)
        # specially recording the corresponding accuracy of class 0 for further validation 
        if ground_truth_label[0] == 0.0:
            accuracies_per_class_iterations_Rest.append([ground_truth_label[0], predict_accu/100])
        
        stop_time_infer = time.time()
        time_infer = stop_time_infer - start_time_infer
        write_inference_time(os.path.join(Online_result_save_rootdir, sub_name), time_infer)

        print("predict accuracy: {}".format(predict_accu))
        print("predict accuracy per class: {}".format(accuracy_per_class))

        if (trial_idx + 1) % update_trial == 0:
            accuracy_per_class_iter = compute_total_accuracy_per_class(accuracies_per_class_iterations)
            accuracy_per_class_iters.append(accuracy_per_class_iter)
            print(accuracy_per_class_iter)
            accuracy_per_class_iter_Rest = compute_total_accuracy_per_class(accuracies_per_class_iterations_Rest)

        # =========================================================================
        # NC论文复现：在线finetune + retrain 双触发机制
        # =========================================================================
        # trial_in_session 已在推理前计算，此处直接使用

        experiment_name = 'lr{}_dropout{}'.format(lr, dropout)
        result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
        result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
        makedir_if_not_exist(result_save_subjectdir)
        makedir_if_not_exist(result_save_subject_checkpointdir)

        # -----------------------------------------------------------------------
        # 触发条件1：session内前6个trial完成 → Mid-session Finetune
        #   对应原论文：用当天前半段（Run 1-8）数据微调模型
        #   冻结 firstConv + depthwiseConv（时域/空域卷积，跨session稳定）
        #   只更新 separableConv + classifier（融合层+分类头，对漂移最敏感）
        #   lr=0.0001，epoch≤100，早停patience=80（监控val_loss），保存val_accuracy最高权重
        # -----------------------------------------------------------------------
        if trial_in_session == FINETUNE_TRIGGER:
            print("******* [NC-Finetune] Mid-session finetune triggered at trial: {} "
                  "(session内第{}/{}个trial) ************".format(
                      trial_idx, trial_in_session, TRIALS_PER_SESSION))
            start_time = time.time()

            # 从当前session的 Base Model 权重出发做finetune（对应原论文逻辑）
            # 原论文finetune时dropout=0.65（大于Orig的0.5），防止小数据集过拟合
            # 需重新实例化模型（以应用新的dropout率），再加载Base Model权重
            if args_dict.model_type in ['EEGNetFea', 'EEGNet']:
                model = EEGNetFea(feature_size=input_feature_size, num_timesteps=512,
                                  num_classes=3, F1=8, D=2, F2=16, dropout=FINETUNE_DROPOUT)
            elif args_dict.model_type in ['DeepConvNetFea', 'DeepConvNet']:
                model = DeepConvNetFea(feature_size=input_feature_size, num_timesteps=512,
                                       num_classes=3, dropout=FINETUNE_DROPOUT)
            elif args_dict.model_type in ['ShallowConvNetFea', 'ShallowConvNet']:
                model = ShallowConvNetFea(feature_size=input_feature_size, num_timesteps=512,
                                          num_classes=3, dropout=FINETUNE_DROPOUT)
            model.load_state_dict(copy.deepcopy(base_model_state_dict))
            model = model.to(device)
            print('[NC-Finetune] Loaded Base Model weights with FINETUNE_DROPOUT={}.'.format(FINETUNE_DROPOUT))

            # 收集当前session前FINETUNE_TRIGGER个trial的在线数据
            # session第1个trial的全局索引 = trial_idx - (FINETUNE_TRIGGER - 1)
            session_start_trial_idx = trial_idx - (FINETUNE_TRIGGER - 1)
            ft_online_features_list = []
            ft_online_labels_list   = []
            for t in range(session_start_trial_idx, trial_idx + 1):
                ft_online_features_list.append(
                    sub_train_feature_array_1[t * batch_size_online : (t + 1) * batch_size_online])
                ft_online_labels_list.append(
                    sub_train_label_array_1[t * batch_size_online : (t + 1) * batch_size_online])
            ft_online_features = np.concatenate(ft_online_features_list, axis=0)
            # shape: (FINETUNE_TRIGGER * batch_size_online, C, T) = (6*9=54, C, T)
            ft_online_labels   = np.concatenate(ft_online_labels_list,   axis=0)

            # 按trial级别8:2随机划分（每trial含batch_size_online=9个segments）
            # 绝不按segment划分，防止同一trial的不同segment分别出现在train和val（数据泄露）
            n_ft_trials = FINETUNE_TRIGGER  # = 6
            shuffled_ft_idx = np.random.permutation(n_ft_trials)
            n_ft_train      = int(0.8 * n_ft_trials)  # 4个trial → train
            ft_train_trial_idx = shuffled_ft_idx[:n_ft_train]
            ft_val_trial_idx   = shuffled_ft_idx[n_ft_train:]

            ft_train_features = np.concatenate(
                [ft_online_features[i * batch_size_online : (i + 1) * batch_size_online]
                 for i in ft_train_trial_idx], axis=0)
            ft_train_labels   = np.concatenate(
                [ft_online_labels[i * batch_size_online : (i + 1) * batch_size_online]
                 for i in ft_train_trial_idx], axis=0)
            ft_val_features   = np.concatenate(
                [ft_online_features[i * batch_size_online : (i + 1) * batch_size_online]
                 for i in ft_val_trial_idx], axis=0)
            ft_val_labels     = np.concatenate(
                [ft_online_labels[i * batch_size_online : (i + 1) * batch_size_online]
                 for i in ft_val_trial_idx], axis=0)

            ft_train_set    = brain_dataset(ft_train_features, ft_train_labels)
            ft_val_set      = brain_dataset(ft_val_features,   ft_val_labels)
            ft_train_loader = torch.utils.data.DataLoader(
                ft_train_set, batch_size=batch_size, shuffle=True)
            ft_val_loader   = torch.utils.data.DataLoader(
                ft_val_set,   batch_size=batch_size, shuffle=False)

            # 冻结 firstConv 和 depthwiseConv，只更新 separableConv 和 classifier
            # EEGNetFea named_children(): firstConv / depthwiseConv / separableConv / classifier
            for name, param in model.named_parameters():
                top_module_name = name.split('.')[0]
                if top_module_name in FREEZE_MODULES:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print('[NC-Finetune] Frozen modules: {}'.format(FREEZE_MODULES))
            print('[NC-Finetune] Trainable param count: {}'.format(
                sum(p.numel() for p in trainable_params)))

            ft_criterion  = nn.CrossEntropyLoss()
            ft_optimizer  = torch.optim.Adam(trainable_params, lr=FINETUNE_LR)
            ft_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                ft_optimizer, mode='min', factor=0.5, patience=30)

            ft_best_val_acc = 0.0
            ft_best_state   = copy.deepcopy(model.state_dict())
            ft_no_improve   = 0  # 监控val_loss连续无改善epoch数（用于早停）
            ft_best_val_loss = float('inf')

            for ft_epoch in range(FINETUNE_EPOCHS):
                # --- train one epoch ---
                model.train()
                for xb, yb in ft_train_loader:
                    xb, yb = xb.to(device), yb.to(device).long()
                    ft_optimizer.zero_grad()
                    out, _ = model(xb)  # EEGNetFea forward返回 (logits, features)
                    loss = ft_criterion(out, yb)
                    loss.backward()
                    ft_optimizer.step()

                # --- validate ---
                model.eval()
                ft_val_correct  = 0
                ft_val_total    = 0
                ft_val_loss_sum = 0.0
                with torch.no_grad():
                    for xb, yb in ft_val_loader:
                        xb, yb = xb.to(device), yb.to(device).long()
                        out, _ = model(xb)
                        loss = ft_criterion(out, yb)
                        ft_val_loss_sum += loss.item() * xb.size(0)
                        preds = out.argmax(dim=1)
                        ft_val_correct += (preds == yb).sum().item()
                        ft_val_total   += xb.size(0)
                ft_val_acc  = ft_val_correct / ft_val_total * 100.0
                ft_val_loss = ft_val_loss_sum / ft_val_total
                ft_scheduler.step(ft_val_loss)

                # 保存val_accuracy最高的权重（原论文：save_best_only=True，监控val_accuracy）
                if ft_val_acc >= ft_best_val_acc:
                    ft_best_val_acc = ft_val_acc
                    ft_best_state   = copy.deepcopy(model.state_dict())

                # 早停监控val_loss（patience=80，原论文设定）
                if ft_val_loss < ft_best_val_loss:
                    ft_best_val_loss = ft_val_loss
                    ft_no_improve    = 0
                else:
                    ft_no_improve += 1
                if ft_no_improve >= FINETUNE_PATIENCE:
                    print('[NC-Finetune] Early stopping at epoch {}, '
                          'best val acc: {:.2f}%'.format(ft_epoch, ft_best_val_acc))
                    break

            # 加载val_accuracy最高的权重（而非最后epoch或早停时的权重）
            model.load_state_dict(ft_best_state)
            finetuned_state_dict = copy.deepcopy(ft_best_state)
            torch.save(finetuned_state_dict,
                       os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
            print('[NC-Finetune] Done. Best val acc: {:.2f}%. Model saved.'.format(
                ft_best_val_acc))

            # 解冻所有层，恢复正常状态供后续推理使用
            # 注意：eval()模式下dropout不激活，无需为推理专门切换dropout率
            # finetuned_state_dict已保存finetune最优权重，后续推理通过load_state_dict使用
            for param in model.parameters():
                param.requires_grad = True
            model = model.to(device)
            model.eval()  # 切回eval模式，dropout在推理时自动关闭

            end_time = time.time()
            write_program_time(os.path.join(Online_result_save_rootdir, sub_name),
                               end_time - start_time)

        # -----------------------------------------------------------------------
        # 触发条件2：一个完整session（12个trial）结束 → Full Retrain
        #   对应原论文：Session结束后用"离线数据全部 + 所有历史在线数据"重训练
        #   全参数更新（不冻结任何层），lr=0.001，epoch≤300，早停patience=80
        #   训练/验证集按trial级别8:2随机划分（在线部分），离线部分全部入训练集
        # -----------------------------------------------------------------------
        if (trial_idx + 1) % TRIALS_PER_SESSION == 0:
            print("******* [NC-Retrain] Full retrain triggered at session end, "
                  "trial: {} ************".format(trial_idx))
            start_time = time.time()

            # combined_feature_array此时 = 离线数据（初始n_offline_segs_init个segments）
            #                             + 所有历史在线trial数据（segment级别，逐trial累积）
            # 在线部分的segment总数 = 已完成trial数 * batch_size_online
            n_online_segs_total = (trial_idx + 1) * batch_size_online
            # 再次确认离线部分大小（与初始化时一致）
            n_offline_segs = combined_feature_array.shape[0] - n_online_segs_total
            # 注意：n_offline_segs 应等于 n_offline_segs_init，此处做断言保护
            assert n_offline_segs == n_offline_segs_init, \
                "[NC-Retrain] n_offline_segs mismatch: {} vs {}".format(
                    n_offline_segs, n_offline_segs_init)

            # 新代码：对应原论文 train_models() 逻辑
            # 离线数据 + 在线数据在 trial 级别混合，整体做 8:2 随机划分，防数据泄露
            # 离线部分：将其视为若干个"trial块"，每块大小同样为 batch_size_online 个 segments
            # 注意：n_offline_segs 可能不能被 batch_size_online 整除，需做整除处理
            offline_feat = combined_feature_array[:n_offline_segs]
            offline_lbl  = combined_label_array[:n_offline_segs]
            online_all_feat = combined_feature_array[n_offline_segs:]
            online_all_lbl  = combined_label_array[n_offline_segs:]

            # 将离线数据按 3 切成 trial 块（尾部不足一块的丢弃，保持 trial 粒度一致）
            OFFLINE_WINDOW_PER_TRIAL = 3  # 离线数据每trial切出3个sliding window
            n_offline_trials = n_offline_segs // OFFLINE_WINDOW_PER_TRIAL
            offline_feat_trimmed = offline_feat[:n_offline_trials * OFFLINE_WINDOW_PER_TRIAL]
            offline_lbl_trimmed  = offline_lbl[:n_offline_trials * OFFLINE_WINDOW_PER_TRIAL]

            # 在线部分 trial 数
            n_online_trials = trial_idx + 1  # 已完成的在线 trial 总数

            # 将离线 trial 块和在线 trial 块统一编号，混合后整体做 8:2 随机划分
            # trial 索引：0 ~ n_offline_trials-1 为离线，n_offline_trials ~ n_total_trials-1 为在线
            n_total_trials = n_offline_trials + n_online_trials
            shuffled_all_idx = np.random.permutation(n_total_trials)
            n_total_train    = int(0.8 * n_total_trials)
            all_train_idx    = shuffled_all_idx[:n_total_train]
            all_val_idx      = shuffled_all_idx[n_total_train:]

            # 改动2：get_segs_by_trial_idx中离线部分步长也用OFFLINE_WINDOW_PER_TRIAL
            def get_segs_by_trial_idx(trial_i):
                if trial_i < n_offline_trials:
                    start = trial_i * OFFLINE_WINDOW_PER_TRIAL  # ← 改这里
                    end   = start + OFFLINE_WINDOW_PER_TRIAL    # ← 改这里
                    return offline_feat_trimmed[start:end], offline_lbl_trimmed[start:end]
                else:
                    online_i = trial_i - n_offline_trials
                    start = online_i * batch_size_online
                    end   = start + batch_size_online
                    return online_all_feat[start:end], online_all_lbl[start:end]

            rt_train_feat = np.concatenate([get_segs_by_trial_idx(i)[0] for i in all_train_idx], axis=0)
            rt_train_lbl  = np.concatenate([get_segs_by_trial_idx(i)[1] for i in all_train_idx], axis=0)
            rt_val_feat   = np.concatenate([get_segs_by_trial_idx(i)[0] for i in all_val_idx],   axis=0)
            rt_val_lbl    = np.concatenate([get_segs_by_trial_idx(i)[1] for i in all_val_idx],   axis=0)

            rt_train_set    = brain_dataset(rt_train_feat, rt_train_lbl)
            rt_val_set      = brain_dataset(rt_val_feat,   rt_val_lbl)
            rt_train_loader = torch.utils.data.DataLoader(
                rt_train_set, batch_size=batch_size, shuffle=True)
            rt_val_loader   = torch.utils.data.DataLoader(
                rt_val_set,   batch_size=batch_size, shuffle=False)

            # 原论文 Orig 模式：从随机初始化出发，dropout固定为0.5（RETRAIN_DROPOUT）
            # 不沿用离线超参搜索的dropout值，以保持与原论文一致
            if preprocess_norm:
                input_feature_size = 30
            else:
                input_feature_size = 29
            if args_dict.model_type in ['EEGNetFea', 'EEGNet']:
                model = EEGNetFea(feature_size=input_feature_size, num_timesteps=512,
                                  num_classes=3, F1=8, D=2, F2=16, dropout=RETRAIN_DROPOUT)
            elif args_dict.model_type in ['DeepConvNetFea', 'DeepConvNet']:
                model = DeepConvNetFea(feature_size=input_feature_size, num_timesteps=512,
                                       num_classes=3, dropout=RETRAIN_DROPOUT)
            elif args_dict.model_type in ['ShallowConvNetFea', 'ShallowConvNet']:
                model = ShallowConvNetFea(feature_size=input_feature_size, num_timesteps=512,
                                          num_classes=3, dropout=RETRAIN_DROPOUT)
            for param in model.parameters():
                param.requires_grad = True
            model = model.to(device)
            print('[NC-Retrain] Model re-initialized from random weights with RETRAIN_DROPOUT={}.'.format(RETRAIN_DROPOUT))

            rt_criterion = nn.CrossEntropyLoss()
            rt_optimizer = torch.optim.Adam(model.parameters(), lr=RETRAIN_LR)
            rt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                rt_optimizer, mode='min', factor=0.5, patience=30)

            rt_best_val_acc  = 0.0
            rt_best_state    = copy.deepcopy(model.state_dict())
            rt_no_improve    = 0  # 监控val_loss连续无改善epoch数（用于早停）
            rt_best_val_loss = float('inf')

            for rt_epoch in trange(RETRAIN_EPOCHS, desc='[NC-Retrain] Full retrain'):
                # --- train one epoch ---
                model.train()
                for xb, yb in rt_train_loader:
                    xb, yb = xb.to(device), yb.to(device).long()
                    rt_optimizer.zero_grad()
                    out, _ = model(xb)  # EEGNetFea forward返回 (logits, features)
                    loss = rt_criterion(out, yb)
                    loss.backward()
                    rt_optimizer.step()

                # --- validate ---
                model.eval()
                rt_val_correct  = 0
                rt_val_total    = 0
                rt_val_loss_sum = 0.0
                with torch.no_grad():
                    for xb, yb in rt_val_loader:
                        xb, yb = xb.to(device), yb.to(device).long()
                        out, _ = model(xb)
                        loss = rt_criterion(out, yb)
                        rt_val_loss_sum += loss.item() * xb.size(0)
                        preds = out.argmax(dim=1)
                        rt_val_correct += (preds == yb).sum().item()
                        rt_val_total   += xb.size(0)
                rt_val_acc  = rt_val_correct / rt_val_total * 100.0
                rt_val_loss = rt_val_loss_sum / rt_val_total
                rt_scheduler.step(rt_val_loss)

                # 保存val_accuracy最高的权重
                if rt_val_acc >= rt_best_val_acc:
                    rt_best_val_acc = rt_val_acc
                    rt_best_state   = copy.deepcopy(model.state_dict())

                # 早停监控val_loss（patience=80）
                if rt_val_loss < rt_best_val_loss:
                    rt_best_val_loss = rt_val_loss
                    rt_no_improve    = 0
                else:
                    rt_no_improve += 1
                if rt_no_improve >= RETRAIN_PATIENCE:
                    print('[NC-Retrain] Early stopping at epoch {}, '
                          'best val acc: {:.2f}%'.format(rt_epoch, rt_best_val_acc))
                    break

            # 加载val_accuracy最高的权重
            model.load_state_dict(rt_best_state)
            # 更新 base_model_state_dict：下一个session的Base Model = 本次retrain最优权重
            base_model_state_dict = copy.deepcopy(rt_best_state)
            # finetuned_state_dict 重置为None，等待下一个session的finetune
            finetuned_state_dict = None
            torch.save(base_model_state_dict,
                       os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
            print('[NC-Retrain] Done. Best val acc: {:.2f}%. Base Model updated for next session.'.format(
                rt_best_val_acc))
            model = model.to(device)
            model.eval()

            end_time = time.time()
            write_program_time(os.path.join(Online_result_save_rootdir, sub_name),
                               end_time - start_time)
        # =========================================================================
    
    accuracy_save2csv(predict_accuracies, result_save_subjectdir, filename='predict_accuracies.csv', columns=['Accuracy'])
    accuracy_save2csv(class_predictions_arrays, result_save_subjectdir, filename='class_predictions_arrays.csv', columns=['class_predictions_arrays'])
    accuracy_save2csv(labels_arrays, result_save_subjectdir, filename='labels_arrays.csv', columns=['labels_arrays'])
    accuracy_save2csv(probabilities_arrays, result_save_subjectdir, filename='probabilities_arrays.csv', columns=['0','1','2'])
    
    plot_calibration_histogram(np.array(labels_arrays), probabilities_arrays, result_save_subjectdir, temperature=2.0, n_bins=10)
    accuracy_iteration_plot(predict_accuracies, result_save_subjectdir)
    accuracy_perclass_save2csv(accuracy_per_class_iters, result_save_subjectdir)
    accuracy_perclass_iteration_plot(accuracy_per_class_iters, result_save_subjectdir)





if __name__ == "__main__":
    
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
    parser.add_argument('--sub_name', default='Jyt', type=str, help='name of the subject')
    parser.add_argument('--windows_num', default=149, type=int, help='number of windows')
    
    parser.add_argument('--Offline_folder_path', default='./Offline_DataCollected', help="Offline folder to the dataset")
    parser.add_argument('--Offline_result_save_rootdir', default='./Offline_experiments', help="Directory containing the experiment models")
    parser.add_argument('--restore_file', default='None', help="xxx.statedict")
    parser.add_argument('--proportion', default=0.8, type=float, help='proportion of the training set of the whole dataset')
    parser.add_argument('--preprocess_norm', default=True, type=str2bool, help="whether to use the BENDR preprocessing")
    parser.add_argument('--n_epoch_offline', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_online', default=100, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=64, type=int, help="number of batch size")
    parser.add_argument('--Online_folder_path', default='./Online_DataCollected', help="Online folder to the dataset")
    parser.add_argument('--Online_result_save_rootdir', default='./Online_experiments', help="Directory containing the experiment models")
    parser.add_argument('--batch_size_online', default=4, type=int, help="number of batch size for online updating")
    parser.add_argument('--trial_pre', default=120, type=int, help="number of samples each class for offline training")
    parser.add_argument('--trial_nums', default=40, type=int, help="number of trails of samples for online updating")
    parser.add_argument('--update_trial', default=1, type=int, help="number of trails for instant updating")
    parser.add_argument('--update_wholeModel', default=12, type=int, help="number of trails for longer updating")
    parser.add_argument('--alpha_distill', default=0.5, type=float, help="alpha of the distillation and cls loss func")
    parser.add_argument('--para_m', default=0.99, type=float, help="hyper parameter for momentum updating")
    parser.add_argument('--cons_rate', default=0.01, type=float, help="hyper parameter for constractive loss")
    parser.add_argument('--best_validation_path', default='lr0.001_dropout0.5', type=str, help="path of the best validation performance model")
    parser.add_argument('--unfreeze_encoder_offline', default=False, type=str2bool, help="whether to unfreeze the encoder params during offline training process")
    parser.add_argument('--unfreeze_encoder_online', default=False, type=str2bool, help="whether to unfreeze the encoder params during online training process")
    parser.add_argument('--T_distil', default=2, type=float, help="the temperature for distillation")
    parser.add_argument('--tau_cons', default=1, type=float, help="the temperature for contrastive loss")
    parser.add_argument('--model_type', default='EEGNet', type=str, help='the base model to use')
    parser.add_argument('--patience', default=20, type=int, help="the patience for early stopping")
    parser.add_argument('--queue_size', default=4*9, type=int, help="the queue size for the TTA method, should be the same as batch_size_online * num_classes")

    parser.add_argument('--ip', default='172.18.22.21', type=str, help='the IP address')
    parser.add_argument('--port', default=8888, type=int, help='the port')
    parser.add_argument('--mode', default='Online', type=str, help='choice of working mode: Offline or Online')
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    sub_name = args.sub_name
    Offline_folder_path  = args.Offline_folder_path
    Online_folder_path = args.Online_folder_path
    windows_num = args.windows_num
    proportion = args.proportion
    Offline_result_save_rootdir = args.Offline_result_save_rootdir
    Online_result_save_rootdir = args.Online_result_save_rootdir
    restore_file = args.restore_file
    n_epoch_offline = args.n_epoch_offline
    n_epoch_online = args.n_epoch_online
    batch_size = args.batch_size
    ip = args.ip
    port = args.port
    mode = args.mode
    batch_size_online = args.batch_size_online
    trial_pre = args.trial_pre
    trial_nums = args.trial_nums
    best_validation_path = args.best_validation_path
    unfreeze_encoder_offline = args.unfreeze_encoder_offline
    unfreeze_encoder_online = args.unfreeze_encoder_online
    update_trial = args.update_trial
    alpha_distill = args.alpha_distill
    update_wholeModel = args.update_wholeModel
    para_m = args.para_m
    cons_rate = args.cons_rate
    preprocess_norm = args.preprocess_norm
    T_distil = args.T_distil
    tau_cons = args.tau_cons
    model_type = args.model_type
    patience = args.patience
    queue_size = args.queue_size

    # save_folder = './Online_DataCollected' + str(sub_name)
    #sanity check:
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))
    print('sub_name: {}, type: {}'.format(sub_name, type(sub_name)))
    print('Offline_folder_path: {}, type: {}'.format(os.path.join(Offline_folder_path, sub_name), type(Offline_folder_path)))
    print('Online_folder_path: {}, type: {}'.format(os.path.join(Online_folder_path, sub_name), type(Offline_folder_path)))
    print('windows_num: {}, type: {}'.format(windows_num, type(windows_num)))
    print('proportion: {}, type: {}'.format(proportion, type(proportion)))
    print('Offline_result_save_rootdir: {}, type: {}'.format(Offline_result_save_rootdir, type(Offline_result_save_rootdir)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch_offline: {} type: {}'.format(n_epoch_offline, type(n_epoch_offline)))
    print('batch size: {} type: {}'.format(batch_size, type(batch_size)))
   
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.sub_name = sub_name
    args_dict.Offline_folder_path = Offline_folder_path
    args_dict.Online_folder_path = os.path.join(Online_folder_path, sub_name)   
    args_dict.windows_num = windows_num
    args_dict.proportion = proportion
    args_dict.Offline_result_save_rootdir = Offline_result_save_rootdir
    args_dict.Online_result_save_rootdir = Online_result_save_rootdir
    args_dict.restore_file = restore_file
    args_dict.n_epoch_offline = n_epoch_offline
    args_dict.n_epoch_online = n_epoch_online
    args_dict.batch_size = batch_size
    args_dict.ip = ip
    args_dict.port = port
    args_dict.mode = mode
    args_dict.batch_size_online = batch_size_online
    args_dict.trial_pre = trial_pre
    args_dict.trial_nums = trial_nums
    args_dict.best_validation_path = best_validation_path
    args_dict.unfreeze_encoder_offline = unfreeze_encoder_offline
    args_dict.unfreeze_encoder_online = unfreeze_encoder_online
    args_dict.accuracy_per_class_init = []
    args_dict.update_trial = update_trial
    args_dict.alpha_distill = alpha_distill
    args_dict.update_wholeModel = update_wholeModel
    args_dict.para_m = para_m
    args_dict.cons_rate = cons_rate
    args_dict.preprocess_norm = preprocess_norm
    args_dict.T_distil = T_distil
    args_dict.tau_cons = tau_cons
    args_dict.model_type = model_type
    args_dict.patience = patience
    args_dict.queue_size = queue_size

    seed_everything(seed)
    if mode == 'offline':
        log_save_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(log_save_dir, 'log_{}.txt'.format(timestamp)), 'w') as f:
            for key, value in args_dict.items():
                f.write('{}: {}\n'.format(key, value))
        # 离线训练模型
        Offline_EEGNet_simulation(args_dict)
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
    
    if mode == 'online':
        log_save_dir = os.path.join(args_dict.Online_result_save_rootdir, args_dict.sub_name)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(log_save_dir, 'log_{}.txt'.format(timestamp)), 'w') as f:
            for key, value in args_dict.items():
                f.write('{}: {}\n'.format(key, value))
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
        
        args_dict.accuracy_per_class_init = best_validation_class_accuracy
        args_dict.restore_file = best_validation_path
        Online_updating_EEGNet_simulation(args_dict)
    
    if mode == 'hybrid':
        log_save_dir = os.path.join(args_dict.Online_result_save_rootdir, args_dict.sub_name)
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(log_save_dir, 'log_{}.txt'.format(timestamp)), 'w') as f:
            for key, value in args_dict.items():
                f.write('{}: {}\n'.format(key, value))
        # 离线训练模型
        Offline_EEGNet_simulation(args_dict)
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
        
        args_dict.accuracy_per_class_init = best_validation_class_accuracy
        args_dict.restore_file = best_validation_path
        Online_updating_EEGNet_simulation(args_dict)

    if mode == 'synthesizing':
        Online_simulation_synthesizing_results_linear(Online_result_save_rootdir=Online_result_save_rootdir)
        #Online_simulation_synthesizing_results_polynomial(Online_result_save_rootdir=Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial_avg(Online_result_save_rootdir=Online_result_save_rootdir)
        #Online_simulation_synthesizing_results_2cls_linear(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_linear_perclass(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial_avgF1(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_calibration_avg(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_calibration_perclass(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial_avgF1_noRest(Online_result_save_rootdir, data_session_avg=24*batch_size_online)
        Online_simulation_synthesizing_results_polynomial_avgF1_Rest(Online_result_save_rootdir, data_session_avg=24*batch_size_online)


    if mode == 'comparison':
        methods = ['baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1_new', 'method5_EEGNet_baseline_1_9batchsize_Rest_2_mixed_3_new', 'method5_EEGNet_baseline_2_7_9batchsize_Rest_2_mixed_3_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_new' ,'method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new']
        methods_perclass = ['method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1_new','method5_EEGNet_baseline_1_9batchsize_Rest_2_mixed_3_new', 'method5_EEGNet_baseline_2_7_9batchsize_Rest_2_mixed_3_new','method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new']
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_1_3', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_2_2', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_2']
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1', 'method5_encoder3_pretrainlight_baseline_1_9batchsize_Rest_2_mixed_2','method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_1', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_2']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1','method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_1']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        #methods = ['method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_1_4', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_3', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_2_2', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_6_mixed_3']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        Online_simulation_synthesizing_results_comparison_linear(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial_optimized(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass(Online_result_save_rootdir, methods_perclass)
        #Online_simulation_synthesizing_results_comparison_linear_2cls(Online_result_save_rootdir, methods)
