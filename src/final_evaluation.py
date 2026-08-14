import torch

from data_loader import test_loader
from model import UNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.3

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/best_unet.pth",
        map_location=DEVICE
    )
)

model.eval()


total_intersection = 0
total_predicted = 0
total_target = 0

positive_dice = []
positive_iou = []
positive_precision = []
positive_recall = []


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        probabilities = torch.sigmoid(
            model(images)
        )

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # Overall pixel-level statistics
        intersection = (
            predictions * masks
        ).sum().item()

        total_intersection += intersection
        total_predicted += predictions.sum().item()
        total_target += masks.sum().item()

        # Positive-slice metrics
        for prediction, target in zip(
            predictions,
            masks
        ):

            if target.sum() == 0:
                continue

            intersection = (
                prediction * target
            ).sum()

            dice = (
                2 * intersection + 1
            ) / (
                prediction.sum()
                + target.sum()
                + 1
            )

            union = (
                prediction.sum()
                + target.sum()
                - intersection
            )

            iou = (
                intersection + 1
            ) / (
                union + 1
            )

            true_positive = intersection

            precision = (
                true_positive + 1
            ) / (
                prediction.sum() + 1
            )

            recall = (
                true_positive + 1
            ) / (
                target.sum() + 1
            )

            positive_dice.append(
                dice.item()
            )

            positive_iou.append(
                iou.item()
            )

            positive_precision.append(
                precision.item()
            )

            positive_recall.append(
                recall.item()
            )


# Overall metrics
overall_dice = (
    2 * total_intersection + 1
) / (
    total_predicted
    + total_target
    + 1
)

overall_iou = (
    total_intersection + 1
) / (
    total_predicted
    + total_target
    - total_intersection
    + 1
)


print("\n===== FINAL TEST RESULTS =====")

print(
    f"Overall Dice:       {overall_dice:.4f}"
)

print(
    f"Overall IoU:        {overall_iou:.4f}"
)

print(
    f"Positive Dice:      "
    f"{sum(positive_dice) / len(positive_dice):.4f}"
)

print(
    f"Positive IoU:       "
    f"{sum(positive_iou) / len(positive_iou):.4f}"
)

print(
    f"Positive Precision:  "
    f"{sum(positive_precision) / len(positive_precision):.4f}"
)

print(
    f"Positive Recall:     "
    f"{sum(positive_recall) / len(positive_recall):.4f}"
)