import torch

from data_loader import val_loader
from model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(DEVICE)
model.load_state_dict(
    torch.load("models/best_unet.pth", map_location=DEVICE)
)
model.eval()


def dice_score(prediction, target, smooth=1.0):
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    return (2 * intersection + smooth) / (denominator + smooth)


thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

scores = {threshold: [] for threshold in thresholds}

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        probabilities = torch.sigmoid(model(images))

        for i in range(images.size(0)):

            # Only evaluate slices containing tumor
            if masks[i].sum() == 0:
                continue

            target = masks[i]

            for threshold in thresholds:

                prediction = (
                    probabilities[i] >= threshold
                ).float()

                dice = dice_score(
                    prediction,
                    target
                )

                scores[threshold].append(
                    dice.item()
                )


for threshold in thresholds:

    if scores[threshold]:

        mean_dice = sum(scores[threshold]) / len(scores[threshold])

        print(
            f"Threshold {threshold:.1f} "
            f"→ Positive-slice Dice {mean_dice:.4f}"
        )