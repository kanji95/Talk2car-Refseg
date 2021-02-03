import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import print_, grad_check
from utils.metrics import compute_mask_IOU, compute_batch_IOU, compute_point_game


def train(
    train_loader,
    joint_model,
    image_encoder,
    optimizer,
    loss_func,
    experiment,
    epochId,
    args,
):

    pid = os.getpid()
    py = psutil.Process(pid)

    joint_model.train()
    image_encoder.eval()

    total_loss = 0
    total_accuracy = 0
    iou_accuracy = 0

    total_inter, total_union = 0, 0

    criterion = nn.CrossEntropyLoss()

    feature_dim = 14

    optimizer.zero_grad()

    epoch_start = time()

    # import pdb; pdb.set_trace()

    data_len = len(train_loader)
    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1

        with torch.no_grad():

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

        loss = loss_func(mask, gt_mask)

        loss.backward()

        if iterId % 500 == 0 and args.grad_check:
            grad_check(joint_model.named_parameters(), experiment)

        optimizer.step()
        joint_model.zero_grad()

        end_time = time()
        elapsed_time = end_time - start_time

        ## import pdb; pdb.set_trace();
        ## mask_mean = mask.mean().item()
        threshold = args.threshold ## min(mask_mean, args.threshold)
        with torch.no_grad():
            inter, union = compute_batch_IOU(mask, gt_mask, threshold)

        total_inter += inter.sum().item()
        total_union += union.sum().item()

        iou_accuracy = total_inter/total_union
        ## iou_score = inter / union
        ## iou_accuracy += (iou_score > 0.5).sum()/batch_size

        total_accuracy += compute_point_game(mask, gt_mask, topk=args.topk)

        total_loss += float(loss.item())

        if iterId % 50 == 0 and step != 0:

            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            curr_loss = total_loss / (step + 1)
            # curr_IOU = total_inter / total_union
            curr_IOU = iou_accuracy / (step + 1)
            curr_acc = total_accuracy / (step + 1)
 
            lr = optimizer.param_groups[0]["lr"]

            print_(
                f"{timestamp} Epoch:[{epochId:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} IOU {curr_IOU:.4f} accuracy {curr_acc:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
            )

    epoch_end = time()
    epoch_time = epoch_end - epoch_start
    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    train_accuracy = total_accuracy / data_len
    total_IOU_acc = iou_accuracy / data_len
    ## overall_IOU = total_inter / total_union

    experiment.log({"loss": train_loss, "IOU": total_IOU_acc, "Accuracy": train_accuracy})

    print_(
        f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} IOU_accuracy {total_IOU_acc:.4f} accuracy {train_accuracy:.4f} elapsed {epoch_time:.2f}"
    )
