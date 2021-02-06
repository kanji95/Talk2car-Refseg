import os
import argparse
from datetime import datetime

import numpy as np
import skimage
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models._utils import IntermediateLayerGetter

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)

from models.modeling.deeplab import *
from dataloader.referit_loader import *
from dataloader.talk2car import Talk2Car

from models.model import JointModel
from utils import im_processing
from utils.utils import log_gpu_usage, print_
from utils.metrics import *


def get_args_parser():
    parser = argparse.ArgumentParser("Refering Image Segmentation", add_help=False)

    # HYPER Params
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

    parser.add_argument("--use_dcrf", default=False, action="store_true")

    # MODEL Params
    parser.add_argument("--image_encoder", type=str, default="deeplabv3_plus")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--skip_conn", default=False, action="store_true")
    parser.add_argument("--model_path", type=str)

    # LOSS Params
    parser.add_argument("--metric", default="pointing_game", type=str, choices=["pointing_game", "intersection_at_t", "recall_at_k", "dice_score"])

    # DATASET parameters
    parser.add_argument(
        "--dataset", type=str, default="referit", choices=["referit", "talk2car"]
    )
    parser.add_argument(
        "--dataroot", type=str, default="/ssd_scratch/cvit/kanishk/referit/"
    )
    parser.add_argument(
        "--glove_path", default="/ssd_scratch/cvit/kanishk/glove/", type=str
    )
    parser.add_argument(
        "--task",
        default="unc",
        type=str,
        choices=["unc", "unc+", "gref", "referit", "talk2car"],
    )
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--cache_type", type=str, default="full")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--image_dim", type=int, default=448)
    parser.add_argument("--mask_dim", type=int, default=56)
    
    parser.add_argument("--mask_thresh", type=float, default=0.40)
    parser.add_argument("--area_thresh", type=float, default=0.50)
    parser.add_argument("--topk", type=int, default=1)

    return parser


def evaluate(image_encoder, joint_model, val_loader, args):

    image_encoder.eval()
    joint_model.eval()

    total_inter = 0
    total_union = 0

    total_accuracy = 0

    total_dcrf_inter, total_dcrf_union = 0, 0

    mean_IOU = 0
    mean_dcrf_IOU = 0

    prec_at_x = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    prec_dcrf_at_x = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}

    data_len = len(val_loader)
    for step, batch in enumerate(val_loader):

        img = batch["image"].cuda(non_blocking=True)

        phrase = batch["phrase"].cuda(non_blocking=True)
        phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)
        index = batch["index"]

        gt_mask = batch["seg_mask"]
        gt_mask = gt_mask.squeeze(dim=1)

        batch_size = img.shape[0]
        img_mask = torch.ones(batch_size, 14 * 14, dtype=torch.int64).cuda(
            non_blocking=True
        )

        with torch.no_grad():
            img = image_encoder(img)

        output_mask = joint_model(img, phrase, img_mask, phrase_mask)
        output_mask = output_mask.detach().cpu()

        if args.use_dcrf:

            orig_image = batch["orig_image"].numpy()
            proc_im = skimage.img_as_ubyte(orig_image)[0]

            H, W = orig_image[0].shape[:-1]

            sigma_val = output_mask

            n_labels = 2

            d = dcrf.DenseCRF2D(H, W, n_labels)

            U = np.expand_dims(-np.log(sigma_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigma_val), axis=0)

            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)

            d.addPairwiseGaussian(sxy=3, compat=5)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)

            Q = d.inference(5)

            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)

            dcrf_output_mask = torch.from_numpy(pred_raw_dcrf).unsqueeze(0)

        inter, union = compute_batch_IOU(output_mask, gt_mask, args.mask_thresh)

        total_inter += inter.sum().item()
        total_union += union.sum().item()

        if args.metric == "pointing_game":
            total_accuracy += pointing_game(output_mask, gt_mask)
        elif args.metric == "intersection_at_t":
            total_accuracy += intersection_at_t(output_mask, gt_mask, args.mask_thresh, args.area_thresh)
        elif args.metric == "recall_at_k":
            total_accuracy += recall_at_k(output_mask, target, args.topk)
        elif args.metric == "dice_score":
            total_accuracy += dice_score(output_mask, gt_mask, args.mask_thresh)


        score = 0 if union.item() == 0 else inter.item() / union.item()

        mean_IOU += score

        total_score = total_inter / total_union

        for x in prec_at_x:
            if score > x:
                prec_at_x[x] += 1

        total_dcrf_score = 0
        if args.use_dcrf:
            dcrf_inter, dcrf_union = compute_mask_IOU(
                dcrf_output_mask, gt_mask, args.mask_thresh
            )

            total_dcrf_inter += dcrf_inter.item()
            total_dcrf_union += dcrf_union.item()

            dcrf_score = dcrf_inter.item() / dcrf_union.item()

            mean_dcrf_IOU += dcrf_score

            total_dcrf_score = total_dcrf_inter / total_dcrf_union

            for x in prec_dcrf_at_x:
                if dcrf_score > x:
                    prec_dcrf_at_x[x] += 1

        if step % 500 == 0:

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            curr_acc = total_accuracy / (step + 1)

            print_(
                f"{timestamp} Step: [{step:5d}/{data_len}] curr_ACC {curr_acc:.5f} IOU {total_score:.5f} dcrf_IOU {total_dcrf_score}"
            )

    overall_IOU = total_inter / total_union
    mean_IOU = mean_IOU / data_len
    final_acc = total_accuracy / data_len

    overall_dcrf_IOU = 0
    if args.use_dcrf:
        overall_dcrf_IOU = total_dcrf_inter / total_dcrf_union
        mean_dcrf_IOU = mean_dcrf_IOU / data_len

    print_(
        f"Final Accuracy {final_acc}, Overall IOU: {overall_IOU}, Mean_IOU: {mean_IOU}, Overall_dcrf_IOU: {overall_dcrf_IOU}, Mean_dcrf_IOU: {mean_dcrf_IOU}"
    )

    for x in prec_at_x:
        percent = (prec_at_x[x] / data_len) * 100
        print_(f"{x}% IOU: {percent}%")

    print_("==================================")

    for x in prec_dcrf_at_x:
        percent = (prec_dcrf_at_x[x] / data_len) * 100
        print_(f"{x}% dcrf_IOU: {percent}%")


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print_(f"{device} being used with {n_gpu} GPUs!!")

    print_("Initializing dataset")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))

    tokenizer = None

    if args.dataset == "referit":
        val_dataset = ReferDataset(
            data_root=args.dataroot,
            dataset=args.task,
            transform=transforms.Compose([resize, to_tensor, normalize]),
            annotation_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
            split=args.split,
            max_query_len=args.seq_len,
            glove_path=args.glove_path,
        )
    elif args.dataset == "talk2car":
        val_dataset = Talk2Car(
            root=args.dataroot,
            split="val",
            transform=transforms.Compose([resize, to_tensor, normalize]),
            mask_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
            glove_path=args.glove_path,
            max_len=args.seq_len,
        )
        args.task = "talk2car"
    else:
        raise NotImplementedError("Dataset not implemented")

    val_loader = DataLoader(
        val_dataset, shuffle=True, batch_size=1, num_workers=1, pin_memory=True
    )

    out_channels = 512
    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    if args.image_encoder == "deeplabv3_plus":
        in_channels = 2048
        stride = 2
        model = DeepLab(num_classes=21, backbone="resnet", output_stride=16)
        model.load_state_dict(
            torch.load("./models/deeplab-resnet.pth.tar")["state_dict"]
        )
        image_encoder = IntermediateLayerGetter(model.backbone, return_layers)
    else:
        raise NotImplemented("Model not implemented")

    for param in image_encoder.parameters():
        param.requires_grad_(False)
    image_encoder.eval()

    vocab_size = 8801 if args.task == "referit" else 12099

    joint_model = JointModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        num_layers=args.num_layers,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        skip_conn=args.skip_conn,
        mask_dim=args.mask_dim,
        vocab_size=vocab_size,
    )

    state_dict = torch.load(args.model_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    joint_model.load_state_dict(state_dict)

    if n_gpu > 1:
        image_encoder = nn.DataParallel(image_encoder)

    joint_model.to(device)
    image_encoder.to(device)

    evaluate(image_encoder, joint_model, val_loader, args)


if __name__ == "__main__":
    main()
