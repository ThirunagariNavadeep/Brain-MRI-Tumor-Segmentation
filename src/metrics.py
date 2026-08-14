import torch

def dice_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()

    return (
        (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    )

def iou_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

def dice_loss(pred, target, smooth=1.0):
    probabilities = torch.sigmoid(pred)

    intersection = (probabilities * target).sum()

    dice = (
        (2 * intersection + smooth) / (probabilities.sum() + target.sum() + smooth)
    )

    return 1 - dice