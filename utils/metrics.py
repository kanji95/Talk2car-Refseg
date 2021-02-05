import torch

from .utils import compute_centroid

@torch.no_grad()
def compute_mask_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = temp.sum()
    union = (((masks > thresh) + target) - temp).sum()
    return intersection, union

@torch.no_grad()
def compute_batch_IOU(masks, target, thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    union = torch.sum(
        (((masks > thresh) + target) - temp).flatten(1), dim=-1, keepdim=True
    )
    mask_area = torch.sum((masks > thresh).flatten(1), dim=-1, keepdim=True)
    return intersection, mask_area


## @torch.no_grad()
## def compute_point_game(masks, target, topk=1):
##     assert target.shape[-2:] == masks.shape[-2:]
##     batch_size = masks.shape[0]
##     # max_indices = masks.flatten(1).argmax(dim=-1)
##     # accuracy = target.flatten(1)[torch.arange(batch_size), max_indices].mean().item()
##     # import pdb; pdb.set_trace();
##     values, indices = torch.topk(masks.flatten(1), k=topk)
##     mean_indices = (values * indices).mean(dim=-1)
##     centroid = torch.stack([mean_indices.floor(), mean_indices.ceil()], dim=-1).long()
##     target_sum = target.flatten(1).gather(1, centroid).sum(dim=-1)
##     accuracy = (target_sum > 0).sum().item()/batch_size
##     return accuracy

@torch.no_grad()
def compute_point_game(masks, target, topk=1):
    assert target.shape[-2:] == masks.shape[-2:]
    batch_size = masks.shape[0]
    # max_indices = masks.flatten(1).argmax(dim=-1)
    # accuracy = target.flatten(1)[torch.arange(batch_size), max_indices].mean().item()
    ## import pdb; pdb.set_trace();
    values, indices = torch.topk(masks.flatten(1), k=topk)
    wt_indices = (values * indices).sum(dim=-1)/values.sum(dim=-1)
    centroid = torch.stack([wt_indices.floor(), wt_indices.ceil()], dim=-1).long()
    target_sum = target.flatten(1).gather(1, centroid).sum(dim=-1)
    accuracy = (target_sum > 0).sum().item()/batch_size
    return accuracy

