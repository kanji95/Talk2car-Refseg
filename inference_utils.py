import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models._utils import IntermediateLayerGetter

from models.modeling.deeplab import *
from dataloader.talk2car import *

from PIL import Image
from skimage.transform import resize

from models.model import JointModel

from utils.im_processing import *
from utils.metrics import *

from collections import Counter
from nltk.corpus import stopwords


class Args:
    def __init__(
        self,
        lr=3e-4,
        num_workers=4,
        image_encoder="deeplabv3_plus",
        num_layers=1,
        num_encoder_layers=1,
        dropout=0.25,
        skip_conn=False,
        model_path="./saved_model/talk2car/baseline_drop_0.25_bs_64_el_1_sl_40_bce_0.50473.pth",
        dataroot="/ssd_scratch/cvit/kanishk/Talk2Car-RefSeg/",
        glove_path="/ssd_scratch/cvit/kanishk/glove/",
        dataset="talk2car",
        task="talk2car",
        split="val",
        seq_len=40,
        image_dim=448,
        mask_dim=448,
        mask_thresh=0.3,
        area_thresh=0.4,
        topk=10,
        metric="pointing_game",
    ):
        self.lr = lr
        self.num_workers = num_workers
        self.image_encoder = image_encoder
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.skip_conn = skip_conn
        self.model_path = model_path
        self.dataroot = dataroot
        self.glove_path = glove_path
        self.dataset = dataset
        self.task = task
        self.split = split
        self.seq_len = seq_len
        self.image_dim = image_dim
        self.mask_dim = mask_dim
        self.mask_thresh = mask_thresh
        self.area_thresh = area_thresh
        self.topk = topk
        self.metric = metric

def compute_mask_IOU(masks, target, thresh=0.3):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = ((masks>thresh) * target)
    intersection = temp.sum()
    union = (((masks>thresh) + target) - temp).sum()
    return intersection, union
        
def meanIOU(m, gt, t):
    temp = ((m > t)*gt)
    inter = temp.sum()
    union = ((m > t) + gt - temp).sum()
    return inter/union

def get_random_sample(val_loader):
    data_len = val_loader.dataset.__len__()
    
    indx = random.choice(range(data_len))
    batch = val_loader.dataset.__getitem__(indx)
    
    return batch

def display_sample(batch):
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(batch["orig_image"])
    plt.title(batch["orig_phrase"])
    plt.axis('off');
    
    plt.show()

def prepare_dataloader(args):
    print("Initializing dataset")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((args.image_dim, args.image_dim))

    val_dataset = Talk2Car(
        root=args.dataroot,
        split=args.split,
        transform=transforms.Compose([resize, to_tensor, normalize]),
        mask_transform=transforms.Compose([ResizeAnnotation(args.mask_dim)]),
        glove_path=args.glove_path,
    )

    val_loader = DataLoader(
        val_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True
    )

    return val_loader

def prepare_network(args, n_gpu, device):
    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    model = DeepLab(num_classes=21, backbone="resnet", output_stride=16)
    model.load_state_dict(torch.load("./models/deeplab-resnet.pth.tar")["state_dict"])

    image_encoder = IntermediateLayerGetter(model.backbone, return_layers)

    for param in image_encoder.parameters():
        param.requires_grad_(False)
        
    in_channels = 2048
    out_channels = 512
    stride = 2

    joint_model = JointModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        num_layers=args.num_layers,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        skip_conn=args.skip_conn,
        mask_dim=args.mask_dim,
    )

    state_dict = torch.load(args.model_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    joint_model.load_state_dict(state_dict) 

    if n_gpu > 1:
        image_encoder = nn.DataParallel(image_encoder)
        joint_model = nn.DataParallel(joint_model)
        
    joint_model.to(device)
    image_encoder.to(device)

    image_encoder.eval();
    joint_model.eval();
    
    return image_encoder, joint_model

def modify_language_command(batch):
    use_original_command = input("Use original command Y/N?: ")
    assert use_original_command in "YNyn"

    original_phrase = True if use_original_command.upper() == "Y" else False

    if not original_phrase:
        new_command = input("Enter New Command: ")
        batch['orig_phrase'] = new_command
    
    return original_phrase

def get_best_threshold(output_mask, gt_mask):    
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

        t_ += 0.05

    if best_t == 0:
        best_t += 0.0001
    return best_t

def run_inference(batch, image_encoder, joint_model, val_loader, args, original_phrase=True):
    
    img = batch["image"].cuda(non_blocking=True).unsqueeze(0)
    
    phrase, phrase_mask = val_loader.dataset.vocabulary.tokenize(batch["orig_phrase"])
    phrase = phrase.unsqueeze(0).cuda(non_blocking=True)
    phrase_mask = phrase_mask.unsqueeze(0).cuda(non_blocking=True)

    gt_mask = batch["seg_mask"]
    gt_mask = gt_mask.squeeze(dim=1)

    orig_image = batch["orig_image"]
    orig_phrase = batch["orig_phrase"]

    batch_size = img.shape[0]
    img_mask = torch.ones(batch_size, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)

    with torch.no_grad():
        img = image_encoder(img)  

    output_mask = joint_model(img, phrase, img_mask, phrase_mask)

    output_mask = output_mask.detach().cpu().squeeze()
    mask_out = output_mask[0]

    inter, union = compute_mask_IOU(output_mask, gt_mask)
    score = inter / union

    image = batch["orig_image"]
    phrase = batch["orig_phrase"]
    mask_gt = gt_mask
    mask_pred = output_mask

    im = image
    
    if original_phrase:
        best_t = get_best_threshold(output_mask, gt_mask)
    else:
        best_t = output_mask.mean()

    ## Prediction
    im_seg = im[:] / 2
    predicts = (mask_pred > best_t).numpy()
    im_seg[:, :, 0] += predicts.astype('uint8') * 100
    im_seg = im_seg.astype('uint8')

    ## Ground Truth
    im_gt = im[:] / 2
    gt = (mask_gt > 0).numpy()
    im_gt[:, :, 1] += gt.astype('uint8') * 100
    im_gt = im_gt.astype('uint8')

    print(f'Command: {phrase}')
    
    if original_phrase:
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

        axes[0].imshow(im_gt)
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        axes[1].imshow(im_seg)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
        
    else:
        
        figure = plt.figure(figsize=(20, 20))
        
        plt.imshow(im_seg)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.show()