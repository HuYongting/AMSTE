import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import random
import logging
from sklearn.metrics import roc_auc_score, roc_curve


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    return 10 * np.log10(1 / mse)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):
    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")  # ???????????[info]?????????
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def chose(list1,list2,th): # 0.2   预测的异常  重构的正常
    for i in range(len(list1)):
        if(list1[i]-list1[i]>=th):
            list1[i] = list2[i]
    return list1

def normalize_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return [item / np.max(item, axis=0) for item in scores]
    else:
        return [(item - np.min(item, axis=0)) / (np.max(item, axis=0) - np.min(item, axis=0)) for item in scores]


def normalize_one_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return scores / np.max(scores, axis=0)
    else:
        return (scores - np.min(scores, axis=0)) / (np.max(scores, axis=0) - np.min(scores, axis=0))


def normalize(sequence_n_frame, scores_appe, scores_flow, scores_comb, scores_angle, ver=2, clip_normalize=True):
    if sequence_n_frame is not None:
        if len(sequence_n_frame) > 1:
            accumulated_n_frame = np.cumsum(sequence_n_frame - 1)[:-1]

            scores_appe = np.split(scores_appe, accumulated_n_frame, axis=0)
            scores_flow = np.split(scores_flow, accumulated_n_frame, axis=0)
            scores_comb = np.split(scores_comb, accumulated_n_frame, axis=0)
            scores_angle = np.split(scores_angle, accumulated_n_frame, axis=0)

            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')
                scores_appe = normalize_clip_scores(scores_appe, ver=ver)
                scores_flow = normalize_clip_scores(scores_flow, ver=ver)
                scores_comb = normalize_clip_scores(scores_comb, ver=ver)
                scores_angle = normalize_clip_scores(scores_angle, ver=ver)

            scores_appe = np.concatenate(scores_appe, axis=0)
            scores_flow = np.concatenate(scores_flow, axis=0)
            scores_comb = np.concatenate(scores_comb, axis=0)
            scores_angle = np.concatenate(scores_angle, axis=0)

        else:
            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')

                scores_appe = np.array(normalize_one_clip_scores(scores_appe, ver=ver))
                scores_flow = np.array(normalize_one_clip_scores(scores_flow, ver=ver))
                scores_comb = np.array(normalize_one_clip_scores(scores_comb, ver=ver))
                scores_angle = np.array(normalize_one_clip_scores(scores_angle, ver=1))

    return scores_appe, scores_flow, scores_angle, scores_comb


def find_max_patch(diff_map_appe, patches=3, size=16, step=4, is_multi=False):
    assert size % step == 0
    # diff_map_appe size: batch * channel * height * width
    b_size = diff_map_appe.shape[0]
    max_mean = np.zeros([b_size, patches])
    std = np.zeros([b_size, patches])
    pos = np.zeros([b_size, patches, 2])

    # sliding window
    for i in range(0, diff_map_appe.shape[-2] - size, step):
        for j in range(0, diff_map_appe.shape[-1] - size, step):
            # mean and std based on patch
            curr_std = np.std(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            curr_mean = np.mean(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            for b in range(b_size):
                for n in range(patches):
                    if curr_mean[b] > max_mean[b, n]:
                        max_mean[b, n + 1:] = max_mean[b, n:-1]
                        std[b, n + 1:] = std[b, n:-1]
                        pos[b, n + 1:] = pos[b, n:-1]
                        max_mean[b, n] = curr_mean[b]
                        std[b, n] = curr_std[b]
                        pos[b, n] = [i, j]
                        break

    if is_multi:
        patches_mean = np.sum(max_mean)
        patches_std = np.sum(std)
        return patches_mean, patches_std
    else:
        return max_mean[:, 0], std[:, 0]

def multi_future_frames_to_scores(input):
    output = cv2.GaussianBlur(input, (5, 0), 10)
    return output

def normalize_score_clip(score, max_score, min_score):
    return ((score - min_score) / (max_score-min_score))


def normalize_score_list_gel(score):           # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(normalize_score_clip(score[i], np.max(score), np.min(score)))
    return anomaly_score_list


def eer(label, score):
    fpr_1, tpr_1, _ = roc_curve(label, score)
    fnr_1 = 1 - tpr_1
    eer = fpr_1[np.nanargmin(np.absolute((fnr_1 - fpr_1)))]
    return eer

def patch_max_mse(diff_map_appe, patches=3, size=16, step=4, is_multi=False):
    assert size % step == 0

    b_size = diff_map_appe.shape[0]
    max_mean = np.zeros([b_size, patches])

    # sliding window
    for i in range(0, diff_map_appe.shape[-2] - size, step):
        for j in range(0, diff_map_appe.shape[-1] - size, step):

            curr_mean = np.mean(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            for b in range(b_size):
                for n in range(patches):
                    if curr_mean[b] > max_mean[b, n]:
                        max_mean[b, n + 1:] = max_mean[b, n:-1]
                        max_mean[b, n] = curr_mean[b]
                        break
    return max_mean[:, 0]  #


def multi_patch_max_mse(diff_map_appe):
    mse_32 = patch_max_mse(diff_map_appe, patches=3, size=32, step=8, is_multi=False)
    mse_64 = patch_max_mse(diff_map_appe, patches=3, size=64, step=16, is_multi=False)
    mse_128 = patch_max_mse(diff_map_appe, patches=3, size=128, step=32, is_multi=False)
    return mse_32,mse_64,mse_128

