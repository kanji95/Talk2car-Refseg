import os
import math
import random
import numpy as np
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

from tqdm import tqdm

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import nvidia_smi

import wandb


def print_(statement, default_gpu=True):
    if default_gpu:
        print(statement, flush=True)

def log_gpu_usage():
    nvidia_smi.nvmlInit()
    print_(f"Driver Version: {nvidia_smi.nvmlSystemGetDriverVersion()}")
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print_(f"mem: {mem_res.used / (1024**2)} (GiB)")  # usage in GiB
        print_(f"mem: {100 * (mem_res.used / mem_res.total):.3f}%")
    nvidia_smi.nvmlShutdown()

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def grad_check(named_parameters, experiment):
    thresh = 0.001

    layers = []
    max_grads = []
    mean_grads = []
    max_colors = []
    mean_colors = []

    for n, p in named_parameters:
        # import pdb; pdb.set_trace()
        # print(n)
        if p.requires_grad and "bias" not in n:
            max_grad = p.grad.abs().max()
            mean_grad = p.grad.abs().mean()
            layers.append(n)
            max_grads.append(max_grad)
            mean_grads.append(mean_grad)

    for i, (val_mx, val_mn) in enumerate(zip(max_grads, mean_grads)):
        if val_mx > thresh:
            max_colors.append("r")
        else:
            max_colors.append("g")
        if val_mn > thresh:
            mean_colors.append("b")
        else:
            mean_colors.append("y")
    ax = plt.subplot(111)
    x = np.arange(len(layers))
    w = 0.3

    ax.bar(x - w, max_grads, width=w, color=max_colors, align="center", hatch="////")
    ax.bar(x, mean_grads, width=w, color=mean_colors, align="center", hatch="----")

    plt.xticks(x - w / 2, layers, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))
    plt.ylim(bottom=0.0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Values")
    plt.title("Model Gradients")

    hatch_dict = {0: "////", 1: "----"}
    legends = []
    for i in range(len(hatch_dict)):
        p = patches.Patch(facecolor="#DCDCDC", hatch=hatch_dict[i])
        legends.append(p)

    ax.legend(legends, ["Max", "Mean"])

    plt.grid(True)
    plt.tight_layout()
    experiment.log({"Gradients": wandb.Image(plt)})
    plt.close()

def meanIOU(m, gt, t):
    temp = ((m > t)*gt)
    inter = temp.sum()
    union = ((m > t) + gt - temp).sum()
    return inter/union

def get_best_IOU(output_mask, gt_mask):
    
    iou = []
    thr = []
    cum_sum = []
    
    t_ = 0.0
    
    best_t = t_
    best_iou = 0
    
    while t_ < 1:
        miou = meanIOU(output_mask, gt_mask, t_)
        cum_sum.append((output_mask > t_).sum())
        iou.append(miou)
        thr.append(t_)
        
        if best_iou < miou:
            best_iou = miou
            best_t = t_
        
        t_ += 0.01
    
    if best_t == 0:
        best_t += 0.05
    
    return best_t

@torch.no_grad()
def log_predicitons(
    orig_image, orig_phrase, output_mask, orig_mask, image_ids, title="train", k=4, threshold=0.4
):
    indices = random.choices(range(output_mask.shape[0]), k=k)

    figure, axes = plt.subplots(nrows=k, ncols=4)
    for i, index in enumerate(indices):
        index = indices[i]

        image_id = image_ids[index].item()

        im = orig_image[index]
        phrase = orig_phrase[index]
        mask_pred = output_mask[index]
        mask_gt = orig_mask[index]

        best_t = get_best_IOU(mask_pred, mask_gt)
        ## print(f'ImageId: {image_id}, Best Threshold: {best_t}')

        im_seg = im[:] / 2
        predicts = (mask_pred > threshold).numpy()
        im_seg[:, :, 0] += predicts.astype('uint8') * 100
        im_seg = im_seg.astype('uint8')

        im_seg_b = im[:] / 2
        predicts_b = (mask_pred > best_t).numpy()
        im_seg_b[:, :, 0] += predicts_b.astype('uint8') * 100
        im_seg_b = im_seg_b.astype('uint8')

        im_gt = im[:] / 2
        gt = (mask_gt > 0).numpy()
        im_gt[:, :, 0] += gt.astype('uint8') * 100
        im_gt = im_gt.astype('uint8')
       
        gt_mask_area = mask_gt.flatten(1).sum().item()

        axes[i, 0].imshow(im_gt)
        axes[i, 0].set_title(f"ground truth ({gt_mask_area})")
        axes[i, 0].set_axis_off()

        phrase_text = '\n'.join(wrap(phrase,50))
        axes[i, 1].imshow(im)
        axes[i, 1].set_title(f'{phrase_text}')
        axes[i, 1].set_axis_off()

        axes[i, 2].imshow(im_seg_b)
        axes[i, 2].set_title(f'best threshold {best_t}')
        axes[i, 2].set_axis_off()

        mask_area = (mask_pred > threshold).sum()
        axes[i, 3].imshow(im_seg)
        axes[i, 3].set_title(f"predicted mask ({image_id}::{mask_area})")
        axes[i, 3].set_axis_off()

    figure.tight_layout()
    wandb.log({f"{title}_segmentation": wandb.Image(figure)}, commit=True)
    plt.close(figure)
