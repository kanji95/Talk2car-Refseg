import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.position_encoding import *

from utils.utils import print_, log_predicitons
from utils.metrics import compute_mask_IOU, compute_batch_IOU, compute_point_game


@torch.no_grad()
def evaluate(
    val_loader, joint_model, image_encoder, loss_func, experiment, epochId, args
):

    image_encoder.eval()
    joint_model.eval()

    pid = os.getpid()
    py = psutil.Process(pid)

    total_loss = 0
    total_accuracy = 0
    iou_accuracy = 0

    total_inter, total_union = 0, 0
    mean_IOU = 0

    n_iter = 0

    feature_dim = 14
    data_len = len(val_loader)

    epoch_start = time()
    for step, batch in enumerate(val_loader):

        img = batch["image"].cuda(non_blocking=True)
        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)

        gt_mask = batch["seg_mask"].cuda(non_blocking=True)
        gt_mask = gt_mask.squeeze(dim=1)

        batch_size = img.shape[0]
        img_mask = torch.ones(
            batch_size, feature_dim * feature_dim, dtype=torch.int64
        ).cuda(non_blocking=True)

        start_time = time()
        with torch.no_grad():
            img = image_encoder(img)
           
        mask = joint_model(img, phrase, img_mask, phrase_mask)
        end_time = time()
        elapsed_time = end_time - start_time

        loss = loss_func(mask, gt_mask)

        ## mask_mean = mask.mean().item()
        threshold = args.threshold ## min(mask_mean, args.threshold)
        inter, union = compute_batch_IOU(mask, gt_mask, threshold)

        total_inter += inter.sum().item()
        total_union += union.sum().item()

        ## import pdb; pdb.set_trace();

        iou_accuracy = total_inter/total_union
        iou_score = inter / union
        ## iou_accuracy += (iou_score > 0.5).sum()/batch_size
        total_accuracy += compute_point_game(mask, gt_mask)

        total_loss += float(loss.item())

        if step % 5 == 0:
            orig_image = batch["orig_image"].numpy()
            orig_phrase = batch["orig_phrase"]
            image_ids = batch["index"]

            ## if epochId % 5 == 0:
            ##     print(f'IOU Score:: {iou_score.view(1, -1)}, Mask_min:: {mask.min().item()}, Mask_max:: {mask.max().item()}')

            log_predicitons(orig_image, orig_phrase, mask.cpu(), gt_mask.cpu(), image_ids, title="val", k=4, threshold=args.threshold)

        if step % 50 == 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            curr_loss = total_loss / (step + 1)
            ## import pdb; pdb.set_trace();
            ## overall_IOU = total_inter / total_union
            curr_IOU = iou_accuracy / (step + 1)
            curr_acc = total_accuracy / (step + 1)

            print_(
                    f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {curr_loss:.4f} IOU_accuracy {curr_IOU:.4f} accuracy {curr_acc:.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
            )

    val_loss = total_loss / data_len
    val_IOU_acc = iou_accuracy / data_len
    val_acc = total_accuracy / data_len

    # val_acc = mean_IOU/data_len

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
    print_(
            f"{timestamp} Validation: EpochId: {epochId:2d} loss {val_loss:.4f} IOU_Accuracy {val_IOU_acc:.4f} accuracy {val_acc:.4f}"
    )
    return val_loss, val_IOU_acc, val_acc
