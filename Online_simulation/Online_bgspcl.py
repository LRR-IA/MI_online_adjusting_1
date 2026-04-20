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

# ── BG-SPCL additions ────────────────────────────────────────────────────────
from scipy.fft import fft, fftfreq
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# BG-SPCL Core Functions
# Choi et al., "Brain-Guided Self-Paced Curriculum Learning for Adaptive
# Human-Machine Interfaces", IEEE TSMC, 2025.
# Source: https://github.com/yeonoi3488/bg-spcl
# ══════════════════════════════════════════════════════════════════════════════

def bg_compute_scores(X, sampling_rate, trial_len):
    """
    Compute normalised Alpha/Theta ratio (ATr) distraction score per EEG trial.
    Adapted from bg_spcl/bg.py: compute_bg_scores().

    Parameters
    ----------
    X            : np.ndarray (N_trials, N_channels, T_timepoints)
    sampling_rate: int   — Hz, e.g. 200
    trial_len    : int   — seconds, e.g. 4  → extracts segments [1s,2s), [2s,3s), [3s,4s)

    Returns
    -------
    scores : np.ndarray (N_trials,) in [0,1]; higher = more distracted = lower quality
    """
    freqs     = fftfreq(sampling_rate, 1.0 / sampling_rate)
    alpha_idx = (freqs >= 8)  & (freqs < 13)
    theta_idx = (freqs >= 4)  & (freqs < 8)

    scores_per_trial = []
    for trial in X:                                        # trial: (C, T)
        seg_scores = []
        for s in range(trial_len - 1):
            seg = trial[:, (s + 1) * sampling_rate : (s + 2) * sampling_rate]
            if seg.shape[1] < sampling_rate:
                continue
            psd         = np.abs(fft(seg)) ** 2           # (C, sampling_rate)
            alpha_power = float(np.mean(psd[:, alpha_idx]))
            theta_power = float(np.mean(psd[:, theta_idx]))
            seg_scores.append(alpha_power / theta_power if theta_power > 1e-10 else 0.0)
        scores_per_trial.append(float(np.mean(seg_scores)) if seg_scores else 0.0)

    scores = np.array(scores_per_trial, dtype=np.float32)
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-10 else np.zeros_like(scores)


def bg_compute_scores_window(X, sampling_rate):
    """
    ATr distraction score for WINDOW-level input.
    Replaces bg_compute_scores when X contains windows (not full trials).
    Computes FFT directly on the entire window instead of 1-second segments.

    X : np.ndarray (N_windows, C, T_window)
    """
    freqs     = fftfreq(X.shape[-1], 1.0 / sampling_rate)
    alpha_idx = (freqs >= 8)  & (freqs < 13)
    theta_idx = (freqs >= 4)  & (freqs < 8)

    scores = []
    for window in X:                            # window: (C, T_window)
        psd         = np.abs(fft(window)) ** 2  # (C, T_window)
        alpha_power = float(np.mean(psd[:, alpha_idx]))
        theta_power = float(np.mean(psd[:, theta_idx]))
        scores.append(alpha_power / theta_power if theta_power > 1e-10 else 0.0)

    scores = np.array(scores, dtype=np.float32)
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-10 else np.zeros_like(scores)

def bg_spcl_offline_train(X, y, model, optimizer, criterion, device,
                          init_lambda=0.125, spcl_lr=0.05, spcl_round=19,
                          k=3.0, sampling_rate=200, trial_len=4):
    """
    One offline BG-SPCL training call (replaces train_one_epoch_fea for the whole epoch).
    Adapted from bg_spcl/spcl.py: main_spcl() + sample_align().

    Steps:
      1. Compute ATr distraction scores.
      2. Tukey masking: discard trials with score > Q3 + k*IQR.
      3. SPL inner loop (spcl_round iterations):
           - Forward pass → proxy difficulty = 1 - p_max * acc_sign
           - Per-class sort ascending (easy first)
           - Select top-λ fraction  (λ = init_lambda + spcl_lr * r, capped at 1.0)
           - Backward on selected samples only

    Parameters
    ----------
    X, y         : np.ndarray (N, C, T) and (N,) — CPU numpy arrays
    model        : nn.Module already on device, must return (logits, features)
    init_lambda  : float — starting fraction of easy samples (paper default 0.125)
    spcl_lr      : float — λ increment per round (paper default 0.05)
    spcl_round   : int   — inner SPL iterations per call (paper default 19)
    k            : float — Tukey fence constant (paper default 3.0)
    sampling_rate: int   — Hz
    trial_len    : int   — seconds
    """
    # ── 1. BG scores ──────────────────────────────────────────────────────────
    scores = bg_compute_scores_window(X, sampling_rate)

    # ── 2. Tukey masking ──────────────────────────────────────────────────────
    Q1, Q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    mask   = scores <= (Q3 + k * (Q3 - Q1))
    print(f'  [BG offline] Tukey removed {(~mask).sum()}/{len(mask)} trials  '
          f'(k={k}, threshold={Q3 + k*(Q3-Q1):.4f})')
    X_m, y_m = X[mask], y[mask]
    if len(X_m) == 0:
        print('  [BG offline] All samples filtered — skipping.')
        return

    X_t = torch.tensor(X_m.astype(np.float32)).to(device)   # (M, C, T)
    y_t = torch.tensor(y_m.astype(np.int64)).to(device)      # (M,)

    # ── 3. SPL inner loop ─────────────────────────────────────────────────────
    model.train()
    for r in range(spcl_round):
        with torch.no_grad():
            logits, _ = model(X_t)
            probs     = torch.softmax(logits, dim=1)         # (M, C)

        p_max, pred = probs.max(dim=1)
        acc_sign    = torch.where(pred == y_t,
                                  torch.ones_like(p_max),
                                  -torch.ones_like(p_max))
        difficulty  = 1.0 - p_max * acc_sign                 # low = easy

        lam = min(init_lambda + spcl_lr * r, 1.0)
        sel = []
        for lbl in torch.unique(y_t).cpu().numpy():
            cls_idx  = (y_t == int(lbl)).nonzero(as_tuple=True)[0]
            order    = torch.argsort(difficulty[cls_idx])    # ascending
            n_pick   = max(1, int(len(cls_idx) * lam))
            sel.append(cls_idx[order[:n_pick]])
        if not sel:
            continue
        sel_idx = torch.cat(sel)

        logits_sel, _ = model(X_t[sel_idx])
        loss = criterion(logits_sel, y_t[sel_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    print(f'  [BG-SPCL offline] {spcl_round} SPL rounds on {len(X_m)} trials.')


def bg_spcl_batch_update(X_buf, model, optimizer, device,
                         num_classes=3, init_lambda=0.125, spcl_lr=0.05,
                         k=3.0, sampling_rate=200, trial_len=4,
                         infer_batch_size=64):
    """
    Online Batch BG-SPCL update — Algorithm 1 lines 11-24 (Choi et al., 2025).
    Missing from the original repo; implemented here.

    Uses SELF-ENTROPY as difficulty (no ground-truth labels):
        H(x) = -1/log(C) * Σ_k δ_k * log(δ_k)
    Low entropy = confident prediction = easy sample (selected first).
    Model minimises self-entropy of selected samples → sharpens predictions.

    GPU memory strategy: gradient accumulation.
      - X_m is kept on CPU at all times.
      - Inference (entropy computation) is done in chunks of infer_batch_size.
      - Backward uses gradient accumulation: loss.sum() over all mini-batches,
        then divide gradients by total count → exactly equivalent to one full-batch
        backward, while keeping peak GPU memory bounded by infer_batch_size.

    Parameters
    ----------
    X_buf           : np.ndarray (Nb, C, T) — memory buffer B (CPU numpy)
    model           : nn.Module on device, returns (logits, features)
    num_classes     : int
    infer_batch_size: int — GPU chunk size. Tune to GPU memory:
                      4GB → 32, 8GB → 64, 12GB → 128
    """
    if len(X_buf) == 0:
        print('  [BG-SPCL online] Empty buffer — skipping.')
        return

    # ── 1. BG masking (CPU only, no GPU memory used) ─────────────────────────
    scores = bg_compute_scores_window(X_buf, sampling_rate)
    Q1, Q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    mask   = scores <= (Q3 + k * (Q3 - Q1))
    print(f'  [BG online] Tukey removed {(~mask).sum()}/{len(mask)} samples.')
    X_m = X_buf[mask]          # stays on CPU throughout
    if len(X_m) == 0:
        print('  [BG-SPCL online] All samples filtered — skipping.')
        return

    print(f'  [BG-SPCL online] {len(X_m)} samples after masking '
          f'(infer_batch_size={infer_batch_size}).')

    log_C = float(np.log(num_classes + 1e-10))

    def batched_entropy(X_np):
        """
        Compute self-entropy H for every sample in X_np.
        Runs inference in chunks of infer_batch_size to bound GPU memory.
        Returns H as a CPU float tensor of shape (N,).
        """
        H_list = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X_np), infer_batch_size):
                chunk = torch.tensor(
                    X_np[start : start + infer_batch_size].astype(np.float32)
                ).to(device)
                logits, _ = model(chunk)
                probs = torch.softmax(logits, dim=1)
                H = -(probs * torch.log(probs + 1e-8)).sum(dim=1) / log_C
                H_list.append(H.cpu())
                del chunk, logits, probs, H        # release GPU memory immediately
        return torch.cat(H_list)                   # (N,) on CPU

    # ── 2. while λ < 1: alternating optimisation ─────────────────────────────
    model.train()
    lam, rnd = init_lambda, 0
    while lam < 1.0:

        # Step a: fix θ → select easy samples by self-entropy (batched, no grad)
        H       = batched_entropy(X_m)                     # (N,) CPU
        n_pick  = max(1, int(len(X_m) * lam))
        _, easy = torch.topk(H, n_pick, largest=False)     # low-H = easy
        easy_np = easy.numpy()                             # CPU numpy indices

        # Step b: fix w* → update θ using gradient accumulation
        # Equivalent to one full-batch backward on all easy samples,
        # but processes infer_batch_size samples at a time to avoid OOM.
        model.train()
        optimizer.zero_grad()
        n_easy = len(easy_np)

        for start in range(0, n_easy, infer_batch_size):
            idx_chunk = easy_np[start : start + infer_batch_size]
            chunk_t   = torch.tensor(
                X_m[idx_chunk].astype(np.float32)).to(device)

            logits_sel, _ = model(chunk_t)
            probs_sel     = torch.softmax(logits_sel, dim=1)
            H_sel = -(probs_sel * torch.log(probs_sel + 1e-8)).sum(dim=1) / log_C

            # Use .sum() (not .mean()) so gradients accumulate correctly.
            # Dividing by n_easy afterwards makes it equivalent to .mean() over
            # the full easy set — i.e. one single full-batch gradient update.
            loss = H_sel.sum() / n_easy
            loss.backward()                    # accumulate gradients, do NOT zero_grad

            del chunk_t, logits_sel, probs_sel, H_sel, loss   # release GPU memory

        optimizer.step()                       # one single parameter update

        # Algorithm 1 line 20 then 19: increment round first, then update λ
        rnd += 1
        lam += rnd * spcl_lr

    model.eval()
    print(f'  [BG-SPCL online batch] {rnd} rounds on {len(X_m)} samples.')


# ══════════════════════════════════════════════════════════════════════════════
# End BG-SPCL Core Functions
# ══════════════════════════════════════════════════════════════════════════════


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
    # ── BG-SPCL offline params ───────────────────────────────────────────────────────
    bg_init_lambda   = getattr(args_dict, 'bg_init_lambda',   0.125)
    bg_spcl_lr       = getattr(args_dict, 'bg_spcl_lr',       0.05)
    bg_spcl_round    = getattr(args_dict, 'bg_spcl_round',    19)
    bg_k             = getattr(args_dict, 'bg_k',             3.0)
    bg_sampling_rate = getattr(args_dict, 'bg_sampling_rate', 200)
    bg_trial_len     = getattr(args_dict, 'bg_trial_len',     4)
    # ─────────────────────────────────────────────────────────────────────────
    
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
                # ── BG-SPCL offline: replaces standard one-epoch train ─────────────
                bg_spcl_offline_train(
                    sub_train_feature_array,   # (N, C, T) numpy, matches your data format
                    sub_train_label_array,     # (N,)      numpy
                    model, optimizer, criterion, device,
                    init_lambda   = bg_init_lambda,
                    spcl_lr       = bg_spcl_lr,
                    spcl_round    = bg_spcl_round,
                    k             = bg_k,
                    sampling_rate = bg_sampling_rate,
                    trial_len     = bg_trial_len,
                )
                average_loss_this_epoch = 0.0  # BG-SPCL updates internally; placeholder for curve logging
                # ─────────────────────────────────────────────────────────────
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
    # ── BG-SPCL online params ────────────────────────────────────────────────────────
    bg_init_lambda   = getattr(args_dict, 'bg_init_lambda',   0.125)
    bg_spcl_lr       = getattr(args_dict, 'bg_spcl_lr',       0.05)
    bg_k             = getattr(args_dict, 'bg_k',             3.0)
    bg_sampling_rate = getattr(args_dict, 'bg_sampling_rate', 200)
    bg_trial_len     = getattr(args_dict, 'bg_trial_len',     4)
    bg_update_trial  = getattr(args_dict, 'bg_update_trial',  update_wholeModel)
 
 
    # ── Algorithm 1, line 2: Initialise memory buffer B with Dtrain ──────────
    # B stores high-confidence pseudo-labelled online samples + replayed Dtrain.
    # Initialised here with offline training data (true labels).
    # nb  : incremental mini-batch size (reuse batch_size_online)
    # r   : memory replay ratio (proportion of nb filled with Dtrain samples)
    # ── Incremental Update 控制参数（对应论文 Algorithm 1 第 4-10 行）────────
    # 忠实还原原仓库设计：
    #   滑动窗口维护最近 _nb_incre 步的 batch 数据（每步 batch_size_online 个窗口）
    #   对窗口内所有样本推理 → 按置信度降序排列 → 取前 _n_online 个作为在线训练样本
    #   同时从 Dtrain 随机取 _n_replay 个真实标注样本做 memory replay
    #   两者混合 → CE 损失更新（对应论文式 11）
    _nb_incre     = 8                        # 滑动窗口大小（trial数），对应论文 nb=8
    _r_replay     = 0.9                      # 离线样本比例，对应论文 r
    _conf_thr     = 0.7                      # 置信度阈值：超过此值才存入 buffer B
    _incre_stride = 8                        # 每步都触发增量更新
    # 滑动窗口总窗口数 = nb个trial × 每个trial的窗口数
    _window_total = _nb_incre * batch_size_online                # = 8*9 = 72 个窗口
    # 在线样本数：从 72 个窗口中取置信度最高的前 window_total*(1-r) 个
    _n_online     = max(1, int(_window_total * (1 - _r_replay))) # = int(72*0.1) = 7
    # 离线样本数：同样以 window_total 为基准，按 r 比例取
    _n_replay     = int(_window_total * _r_replay)               # = int(72*0.9) = 64
    # Note: _n_replay will be capped after sub_train_label_array is loaded below
    # 滑动窗口 list，最多保存 _nb_incre 个 trial 的 batch
    _sliding_window_X = []                   # 每个元素 shape: (batch_size_online, C, T)
    # ─────────────────────────────────────────────────────────────────────────

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
    # Cap _n_replay to Dtrain size now that sub_train_label_array is defined
    _n_replay = min(_n_replay, len(sub_train_label_array))
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
    
    # using a FIFO queue to save the latest queue_size num of data for model updating for TTA method
    feature_queue = deque(maxlen=queue_size)

    # ── Algorithm 1, line 2: Initialise memory buffer B with Dtrain ──────────
    memory_B_X = sub_train_feature_array.copy()   # (N_train, C, T)  true labels
    memory_B_y = sub_train_label_array.copy()     # (N_train,)
    # ─────────────────────────────────────────────────────────────────────────
    # Initialise optimizer before the online loop (used by both Incremental and Batch Update)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # this is the implementation of the 
    # paper: 
    for trial_idx in range(trial_nums):
        # generate the new data, simulating the online experiment 
        sub_train_feature_batches = sub_train_feature_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online, :, :]
        sub_train_label_batches = sub_train_label_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online]
        
        # Add data to the FIFO queue (kept for compatibility)
        feature_queue.append(sub_train_feature_batches)

        # online simulation trials
        print("********** Online simulation trial: {} ***********".format(trial_idx))
        start_time_infer = time.time()

        model = model.to(device)
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

        # ── Algorithm 1, lines 4-10: Incremental Update ───────────────────────
        # Design: faithful to original bg-spcl repo (sliding window + confidence selection)
        #   Step 1: maintain a sliding window of the last _nb_incre batches
        #   Step 2: run inference on ALL samples in the window → rank by confidence
        #   Step 3: top-_n_online high-confidence → online training samples (pseudo-labels)
        #           also store them to buffer B if conf > _conf_thr  (Line 7, for Batch Update)
        #   Step 4: randomly sample _n_replay from Dtrain → replay samples (true labels)
        #   Step 5: combine online + replay → CE loss → optimizer.step()  (Eq. 11)

        # Update sliding window (keep last _nb_incre batches)
        _sliding_window_X.append(sub_train_feature_batches.copy())
        if len(_sliding_window_X) > _nb_incre:
            _sliding_window_X.pop(0)

        if (trial_idx + 1) % _incre_stride == 0 and len(_sliding_window_X) == _nb_incre:

            # Concatenate sliding window → (nb * batch_size_online, C, T)
            X_window    = np.concatenate(_sliding_window_X, axis=0)
            X_window_t  = torch.tensor(X_window.astype(np.float32)).to(device)

            # Line 6: run inference on all window samples → confidence + pseudo-labels
            model.eval()
            with torch.no_grad():
                logits_win, _ = model(X_window_t)
                probs_win     = torch.softmax(logits_win, dim=1)
                conf_win, pseudo_win = probs_win.max(dim=1)   # (nb*batch_size_online,)

            # Line 7: store HIGH-CONFIDENCE samples into buffer B
            # B is used by Batch Update (lines 11-24), NOT directly for training here.
            high_conf_mask = conf_win.cpu().numpy() > _conf_thr
            if high_conf_mask.sum() > 0:
                memory_B_X = np.concatenate(
                    [memory_B_X, X_window[high_conf_mask]], axis=0)
                memory_B_y = np.concatenate(
                    [memory_B_y, pseudo_win.cpu().numpy()[high_conf_mask]], axis=0)

            # Select top-_n_online samples by confidence (descending) as online portion
            # This mirrors the original repo's: indices[:int(buffer_size - replay_ratio)]
            _, top_conf_idx = torch.topk(conf_win, _n_online, largest=True)
            top_conf_idx_np = top_conf_idx.cpu().numpy()
            X_online_sel    = X_window[top_conf_idx_np]                    # (_n_online, C, T)
            y_online_sel    = pseudo_win.cpu().numpy()[top_conf_idx_np]    # (_n_online,)

            # Line 8: randomly sample _n_replay labeled examples from Dtrain (memory replay)
            r_idx    = np.random.choice(
                len(sub_train_label_array), _n_replay, replace=True)
            X_replay = sub_train_feature_array[r_idx]   # (_n_replay, C, T)  true labels
            y_replay = sub_train_label_array[r_idx]     # (_n_replay,)

            # Line 9: combined batch = online (pseudo) + replay (true) → Eq. (11) CE loss
            X_incre   = np.concatenate([X_online_sel, X_replay], axis=0)
            y_incre   = np.concatenate([y_online_sel, y_replay],  axis=0)
            X_incre_t = torch.tensor(X_incre.astype(np.float32)).to(device)
            y_incre_t = torch.tensor(y_incre.astype(np.int64)).to(device)

            model.train()
            optimizer.zero_grad()                        # reuse outer optimizer (preserve Adam momentum)
            logits_incre, _ = model(X_incre_t)
            loss_incre = nn.CrossEntropyLoss()(logits_incre, y_incre_t)
            loss_incre.backward()
            optimizer.step()
            model.eval()

            print(f'  [BG-SPCL incre] trial {trial_idx}: '
                  f'window={len(X_window)}, online(top-conf)={_n_online}, '
                  f'replay={_n_replay}, '
                  f'loss={loss_incre.item():.4f}, buf_B={len(memory_B_y)}')
        # ── end Incremental Update ─────────────────────────────────────────────

        # ── BG-SPCL Scheme C: online batch update (Algorithm 1, lines 11-24) ──
        # Triggered every bg_update_trial trials (aligned with update_wholeModel).
        if (trial_idx + 1) % bg_update_trial == 0:
            print(f'******* BG-SPCL online batch update  trial: {trial_idx} *******')
            start_time = time.time()

            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)
            result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)

            model = model.to(device)

            # Algorithm 1, line 13: use memory buffer B (not raw session data)
            # B contains high-confidence pseudo-labelled online samples
            # plus replayed Dtrain, giving a cleaner curriculum signal.
            X_buf = memory_B_X.copy()

            bg_spcl_batch_update(
                X_buf, model, optimizer, device,
                num_classes      = 3,           # change if your task has ≠ 3 classes
                init_lambda      = bg_init_lambda,
                spcl_lr          = bg_spcl_lr,
                k                = bg_k,
                sampling_rate    = bg_sampling_rate,
                trial_len        = bg_trial_len,
                infer_batch_size = 64,          # 4GB GPU→32, 8GB→64, 12GB→128
            )

            # Save updated model
            torch.save(model.state_dict(),
                       os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))

            end_time = time.time()
            print(f'  BG-SPCL batch update finished in {end_time - start_time:.1f}s')
        # ─────────────────────────────────────────────────────────────────────────
    
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
    parser.add_argument('--update_trial', default=15, type=int, help="number of trails for instant updating")
    parser.add_argument('--update_wholeModel', default=15, type=int, help="number of trails for longer updating")
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
    # ── BG-SPCL hyperparameters ──────────────────────────────────────────────
    parser.add_argument('--bg_init_lambda',   default=0.125, type=float,
                        help='BG-SPCL: initial easy-sample fraction per SPL round')
    parser.add_argument('--bg_spcl_lr',       default=0.05,  type=float,
                        help='BG-SPCL: lambda increment per SPL round (delta_lambda)')
    parser.add_argument('--bg_spcl_round',    default=19,    type=int,
                        help='BG-SPCL: number of SPL iterations per offline epoch call')
    parser.add_argument('--bg_k',             default=3.0,   type=float,
                        help='BG-SPCL: Tukey fence constant for distraction filtering')
    parser.add_argument('--bg_sampling_rate', default=200,   type=int,
                        help='BG-SPCL: EEG sampling rate in Hz')
    parser.add_argument('--bg_trial_len',     default=4,     type=int,
                        help='BG-SPCL: trial length in seconds')
    parser.add_argument('--bg_update_trial',  default=12,    type=int,
                        help='BG-SPCL: online batch update every N trials (aligned with update_wholeModel)')
    # ─────────────────────────────────────────────────────────────────────────

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
    # ── BG-SPCL ──────────────────────────────────────────────────────────────
    bg_init_lambda   = args.bg_init_lambda
    bg_spcl_lr       = args.bg_spcl_lr
    bg_spcl_round    = args.bg_spcl_round
    bg_k             = args.bg_k
    bg_sampling_rate = args.bg_sampling_rate
    bg_trial_len     = args.bg_trial_len
    bg_update_trial  = args.bg_update_trial
    # ─────────────────────────────────────────────────────────────────────────

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
    # ── BG-SPCL ──────────────────────────────────────────────────────────────
    args_dict.bg_init_lambda   = bg_init_lambda
    args_dict.bg_spcl_lr       = bg_spcl_lr
    args_dict.bg_spcl_round    = bg_spcl_round
    args_dict.bg_k             = bg_k
    args_dict.bg_sampling_rate = bg_sampling_rate
    args_dict.bg_trial_len     = bg_trial_len
    args_dict.bg_update_trial  = bg_update_trial
    # ─────────────────────────────────────────────────────────────────────────

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
