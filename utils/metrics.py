import torch

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
    return intersection, union # mask_area


@torch.no_grad()
def compute_point_game(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    batch_size = masks.shape[0]
    max_indices = masks.flatten(1).argmax(dim=-1)
    accuracy = target.flatten(1)[torch.arange(batch_size), max_indices].mean().item()
    return accuracy
