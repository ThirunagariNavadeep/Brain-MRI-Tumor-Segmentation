from pathlib import Path

from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from split import get_patient_splits


DATASET_DIR = Path("Datasets/lgg-mri-segmentation/kaggle_3m")

train_images, val_images, test_images = get_patient_splits(DATASET_DIR)

train_dataset = BrainMRIDataset(train_images)
val_dataset = BrainMRIDataset(val_images)
test_dataset = BrainMRIDataset(test_images)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)