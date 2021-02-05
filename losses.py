import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.utils import compute_centroid

class Dice_loss:
    def __call__(self, inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        assert -1 not in denominator
        loss = 1 - (numerator + 1) / (denominator + 1)
        assert not torch.any(torch.isnan(loss))
        return loss.sum()

class Kl_Divergence:
    
    #@torchsnooper.snoop()
    def __call__(self, s_map, gt):

        batch_size = s_map.size(0)
            
        w = s_map.size(1)
        h = s_map.size(2)

        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
            
        assert expand_s_map.size() == s_map.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
            
        assert expand_gt.size() == gt.size()
        
        s_map = s_map/(expand_s_map*1.0)
        assert torch.isnan(s_map).sum().item() == 0

        gt = gt / (expand_gt*1.0)
        assert torch.isnan(gt).sum().item() == 0

        s_map = s_map.view(batch_size, -1)
        gt = gt.view(batch_size, -1)

        eps = 2.2204e-16
        result = gt * torch.log(eps + gt/(s_map + eps))
        
        return torch.mean(torch.sum(result, 1))

def weighted_bce(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

class Loss:
    def __init__(self, args):

        self.args = args
        self.topk = args.topk
        self.mask_dim = args.mask_dim

        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss()
        self.kl_div = Kl_Divergence()
        self.dice_loss = Dice_loss()

    def __call__(self, inputs, targets):

        # import pdb; pdb.set_trace()
        loss = 0
        if "dice" in self.args.loss:
            loss += 0.3*self.dice_loss(inputs, targets)
        if "kl_div" in self.args.loss:
            loss += self.kl_div(inputs, targets)
        if "l1" in self.args.loss:
            loss += self.l1_loss(inputs, targets)
        if "centl" in self.args.loss:
            centroid = compute_centroid(inputs, self.args.topk)

            gt_indices = torch.arange(self.args.mask_dim*self.args.mask_dim).cuda()
            gt_centroid = (targets.flatten(1)*gt_indices).sum(-1)/targets.flatten(1).sum(-1)

            centroid = centroid.round()
            gt_centroid = gt_centroid.round()

            cent_x, cent_y = centroid/self.mask_dim, centroid % self.mask_dim
            cent_x, cent_y = cent_x/self.mask_dim, cent_y/self.mask_dim

            gt_cent_x, gt_cent_y = gt_centroid/self.mask_dim, gt_centroid % self.mask_dim
            gt_cent_x, gt_cent_y = gt_cent_x/self.mask_dim, gt_cent_y/self.mask_dim

            loss += 0.7*(self.mse_loss(cent_x, gt_cent_x) + self.mse_loss(cent_y, gt_cent_y))
        if "bce" in self.args.loss:
            inputs = inputs.flatten(1)
            targets = targets.flatten(1)
            _, indices = torch.topk(inputs, k=self.topk)
            input_values = inputs.gather(1, indices)
            target_values = targets.gather(1, indices)
            loss += self.bce_loss(input_values, target_values)
            ## loss += self.bce_loss(inputs[targets == 1], targets[targets == 1])
            ## inputs = torch.clamp(inputs, 1e-5, 0.99)
            ## loss += (-0.75*targets*torch.log(inputs) -0.25*(1 - targets)*torch.log(1 - inputs)).mean()
            ## loss += weighted_bce(inputs, targets, [0.75, 0.25])
        if "mse" in self.args.loss:
            loss += self.mse_loss(inputs, targets)
        if loss == 0:
            raise Exception(f"{self.args.loss} loss not implemented!")
        return loss
