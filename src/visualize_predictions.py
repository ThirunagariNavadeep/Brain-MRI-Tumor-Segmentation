import matplotlib.pyplot as plt
import torch

from data_loader import test_loader
from model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(DEVICE)
model.load_state_dict(
    torch.load("models/best_unet.pth", map_location=DEVICE)
)
model.eval()

positive_samples = []

with torch.no_grad():
    for images, masks in test_loader:

        # Move MRI images to the same device as the model
        images = images.to(DEVICE)

        predictions = torch.sigmoid(model(images))

        for i in range(images.size(0)):

            # Keep only tumor-positive test slices
            if masks[i].sum() > 0:

                positive_samples.append(
                    (
                        images[i, 0].cpu(),
                        masks[i, 0].cpu(),
                        predictions[i, 0].cpu(),
                    )
                )

            if len(positive_samples) == 3:
                break

        if len(positive_samples) == 3:
            break


if len(positive_samples) == 0:
    raise RuntimeError("No tumor-positive samples found in the test set.")


fig, axes = plt.subplots(
    len(positive_samples),
    4,
    figsize=(12, 9)
)

# Handle the case where only one sample is found
if len(positive_samples) == 1:
    axes = axes.reshape(1, 4)


for i, (image, mask, prediction) in enumerate(positive_samples):

    # Convert probability map to binary mask
    prediction = (prediction > 0.5).float()

    # MRI
    axes[i, 0].imshow(image, cmap="gray")
    axes[i, 0].set_title("MRI")
    axes[i, 0].axis("off")

    # Ground truth
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    # Prediction
    axes[i, 2].imshow(prediction, cmap="gray")
    axes[i, 2].set_title("Prediction")
    axes[i, 2].axis("off")

    # Overlay
    axes[i, 3].imshow(image, cmap="gray")
    axes[i, 3].imshow(prediction, alpha=0.4)
    axes[i, 3].set_title("Prediction Overlay")
    axes[i, 3].axis("off")


plt.tight_layout()

plt.savefig(
    "results/positive_test_predictions.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()