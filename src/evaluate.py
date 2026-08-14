import torch
from data_loader import test_loader
from model import UNet
from metrics import dice_score, iou_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(DEVICE)
model.load_state_dict(torch.load("models/best_unet.pth", map_location = DEVICE))
model.eval()

total_dice = 0.0 
total_iou = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        predictions = model(images)

        total_dice += dice_score(predictions, masks).item()
        total_iou += iou_score(predictions, masks).item()

dice = total_dice / len(test_loader)
iou = total_iou / len(test_loader)

print(f"Test Dice: {dice:.4f}")
print(f"Test IoU:  {iou:.4f}")