import wandb

import argparse
import gc
import os
import random
import traceback
from datetime import datetime
from time import time
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

## import torch.multiprocessing as mp
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler

from torchvision.models._utils import IntermediateLayerGetter

from models.modeling.deeplab import *

from dataloader.referit_loader import ReferDataset
from dataloader.talk2car import Talk2Car

from evaluate import evaluate
from losses import Loss
from models.model import JointModel
from train import train
from utils.utils import print_

plt.rcParams["figure.figsize"] = (15, 15)

torch.manual_seed(12)
torch.cuda.manual_seed(12)
random.seed(12)
np.random.seed(12)

torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out


def get_args_parser():
    parser = argparse.ArgumentParser("Refering Image Segmentation", add_help=False)

    # HYPER Params
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--power_factor", default=0.9, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--grad_check", default=False, action="store_true")

    ## DETECTRON Predictor
    parser.add_argument("--predictor", default=False, action="store_true")

    ## DCRF
    parser.add_argument("--dcrf", default=False, action="store_true")

    # MODEL Params
    parser.add_argument(
        "--image_encoder",
        type=str,
        default="deeplabv3_plus",
        choices=[
            "vgg16",
            "vgg19",
            "resnet50",
            "resnet101",
            "deeplabv2",
            "deeplabv3_resnet101",
            "deeplabv3_plus",
        ],
    )
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--skip_conn", default=False, action="store_true")

    parser.add_argument("--model_dir", type=str, default="./saved_model")
    parser.add_argument("--save", default=False, action="store_true")

    parser.add_argument("--model_filename", default="model_talk2car.pth", type=str)

    # LOSS Params
    parser.add_argument("--loss", default="bce", type=str)

    parser.add_argument("--run_name", default="", type=str)

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
        default="talk2car",
        type=str,
        choices=[
            "talk2car",
            "unc",
            "unc+",
            "gref",
            "referit",
        ],
    )
    parser.add_argument("--cache_type", type=str, default="full")
    parser.add_argument("--image_dim", type=int, default=448)
    parser.add_argument("--mask_dim", type=int, default=448)
    parser.add_argument("--seq_len", type=int, default=20)

    parser.add_argument("--mask_thresh", type=float, default=0.40)
    parser.add_argument("--area_thresh", type=float, default=0.50)
    parser.add_argument("--topk", type=int, default=1)

    return parser


def main(args):

    experiment = wandb.init(project="vigil-network", config=args)
    if args.run_name == "":
        print_("No Name Provided, Using Default Run Name")
        args.run_name = f"{args.task}-{experiment.id}"
    print_(f"METHOD USED FOR CURRENT RUN {args.run_name}")
    experiment.name = args.run_name
    wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print_(f"{device} being used with {n_gpu} GPUs!!")

    ####################### Model Initialization #######################

    out_channels = 512
    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    if "vgg" in args.image_encoder:
        in_channels = 512
        stride = 1
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = model.features
    elif args.image_encoder == "resnet50" or args.image_encoder == "resnet101":
        in_channels = 2048
        stride = 1
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = IntermediateLayerGetter(model, return_layers)
    elif args.image_encoder == "deeplabv2":
        in_channels = 2048
        stride = 2
        model = torch.hub.load(
            "kazuto1011/deeplab-pytorch",
            "deeplabv2_resnet101",
            pretrained="voc12",
            n_classes=21,
        )
        return_layers = {"layer3": "layer2", "layer4": "layer3", "layer5": "layer4"}
        image_encoder = IntermediateLayerGetter(model.base, return_layers)
    elif args.image_encoder == "deeplabv3_resnet101":
        in_channels = 2048
        stride = 2
        model = torch.hub.load(
            "pytorch/vision:v0.6.1", args.image_encoder, pretrained=True
        )
        image_encoder = IntermediateLayerGetter(model.backbone, return_layers)
    elif args.image_encoder == "deeplabv3_plus":
        in_channels = 2048
        stride = 2
        model = DeepLab(num_classes=21, backbone="resnet", output_stride=16)
        model.load_state_dict(torch.load("./models/deeplab-resnet.pth.tar")["state_dict"])
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

    wandb.watch(joint_model, log="all")

    if n_gpu > 1:
        image_encoder = nn.DataParallel(image_encoder)

    joint_model.to(device)
    image_encoder.to(device)

    total_parameters = 0
    for name, child in joint_model.named_children():
        num_params = sum([p.numel() for p in child.parameters() if p.requires_grad])
        if num_params > 0:
            print_(f"No. of params in {name}: {num_params}")
            total_parameters += num_params

    print_(f"Total number of params: {total_parameters}")

    ## Initialize Optimizers
    param_dicts = [
        {"params": [p for p in joint_model.parameters() if p.requires_grad]},
        ## {"params": [p for p in image_encoder.parameters() if p.requires_grad]},
    ]

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    # Loss Calculator
    loss_func = Loss(args)

    if args.dataset == "talk2car":
        args.task = "talk2car"

    save_path = os.path.join(args.model_dir, args.task)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_filename = os.path.join(
        save_path,
        f'{args.image_encoder}_{datetime.now().strftime("%d_%b_%H-%M")}.pth',
    )

    ######################## Dataset Loading ########################
    print_("Initializing dataset")
    start = time()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))
    random_grayscale = transforms.RandomGrayscale(p=0.3)

    if args.dataset == "referit":
        train_dataset = ReferDataset(
            data_root=args.dataroot,
            dataset=args.task,
            transform=transforms.Compose(
                [resize, random_grayscale, to_tensor, normalize]
            ),
            annotation_transform=transforms.Compose(
                [ResizeAnnotation(args.mask_dim)]
            ),
            split="train",
            max_query_len=args.seq_len,
            glove_path=args.glove_path,
        )
        val_dataset = ReferDataset(
            data_root=args.dataroot,
            dataset=args.task,
            transform=transforms.Compose([resize, to_tensor, normalize]),
            annotation_transform=transforms.Compose(
                [ResizeAnnotation(args.mask_dim)]
            ),
            split="val",
            max_query_len=args.seq_len,
            glove_path=args.glove_path,
        )
    elif args.dataset == "talk2car":
        train_dataset = Talk2Car(
            root=args.dataroot,
            split="train",
            transform=transforms.Compose(
                [resize, random_grayscale, to_tensor, normalize]
            ),
            mask_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
            glove_path=args.glove_path,
            max_len=args.seq_len,
        )
        ## train_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [8000, 349])
        val_dataset = Talk2Car(
            root=args.dataroot,
            split="val",
            transform=transforms.Compose([resize, to_tensor, normalize]),
            mask_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
            glove_path=args.glove_path,
            max_len=args.seq_len,
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataset is {elapsed}sec")

    start = time()

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataloader is {elapsed}sec")

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.power_factor,
        patience=2,
        threshold=1e-3,
        min_lr=1e-8,
        verbose=True,
    )

    num_iter = len(train_loader)
    print_(f"training iterations {num_iter}")

    print_(
        f"===================== SAVING MODEL TO FILE {model_filename}! ====================="
    )

    best_acc = 0
    epochs_without_improvement = 0

    for epochId in range(args.epochs):

        train(
            train_loader,
            joint_model,
            image_encoder,
            optimizer,
            loss_func,
            experiment,
            epochId,
            args,
        )

        val_loss, val_IOU, val_acc = evaluate(
            val_loader,
            joint_model,
            image_encoder,
            loss_func,
            experiment,
            epochId,
            args,
        )

        wandb.log({"val_loss": val_loss, "val_IOU": val_IOU, "val_acc": val_acc})

        lr_scheduler.step(val_loss)

        if val_acc > best_acc and args.save:
            best_acc = val_acc
            print_(
                f"Saving Checkpoint at epoch {epochId}, best validation accuracy is {best_acc}!"
            )
            torch.save(
                {
                    "epoch": epochId,
                    "state_dict": joint_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_filename,
            )
            epochs_without_improvement = 0
        elif val_acc <= best_acc:
            epochs_without_improvement += 1
            print_(f"Epochs without Improvement: {epochs_without_improvement}")

            if epochs_without_improvement == 8 and epochId != args.epochs - 1:
                print_(
                    f"{epochs_without_improvement} epochs without improvement, Stopping Training!"
                )
                break
    if args.save:
        print_(f"Current Run Name {args.run_name}")
        best_acc_filename = os.path.join(
            save_path,
            f"{args.image_encoder}_{args.dataset}_{args.num_encoder_layers}_{args.loss}_{best_acc:.5f}.pth",
        )
        os.rename(model_filename, best_acc_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Referring Image Segmentation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    print_(args)

    try:
        main(args)
    except Exception as e:
        traceback.print_exc()
