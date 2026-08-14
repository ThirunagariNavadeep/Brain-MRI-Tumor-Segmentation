import torch
import torch.nn as nn
from torch.optim import Adam

from data_loader import train_loader, val_loader
from model import UNet
from metrics import dice_score, iou_score, dice_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
LR = 1e-4
PATIENCE = 4
POS_WEIGHT = 94.2854

model = UNet().to(DEVICE)

pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = Adam(model.parameters(), lr=LR)


def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)


best_dice = 0.0
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_fn(predictions, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            val_loss += loss_fn(
                predictions,
                masks
            ).item()

            val_dice += dice_score(
                predictions,
                masks
            ).item()

            val_iou += iou_score(
                predictions,
                masks
            ).item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Dice: {val_dice:.4f} "
        f"Val IoU: {val_iou:.4f}"
    )

    if val_dice > best_dice:

        best_dice = val_dice
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "models/best_unet_weighted.pth"
        )

        print("  → Best weighted model saved.")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("Early stopping.")
        break


print(f"\nBest validation Dice: {best_dice:.4f}")

