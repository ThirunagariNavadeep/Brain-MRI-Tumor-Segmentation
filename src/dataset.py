from pathlib import Path
from preprocessing import preprocess_image
import cv2
import torch
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = preprocess_image(image)
        mask = (mask > 0).astype("float32")

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask