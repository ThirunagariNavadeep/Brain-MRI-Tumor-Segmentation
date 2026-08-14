from pathlib import Path
import random


def get_patient_splits(dataset_dir, seed=42):
    images = [
        p for p in Path(dataset_dir).rglob("*.tif")
        if "_mask" not in p.stem
    ]

    patients = {}

    for image_path in images:
        patient_id = image_path.parent.name
        patients.setdefault(patient_id, []).append(image_path)

    patient_ids = sorted(patients)

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    train_patients = set(patient_ids[:train_end])
    val_patients = set(patient_ids[train_end:val_end])
    test_patients = set(patient_ids[val_end:])

    train_images = [
        p for patient in train_patients
        for p in patients[patient]
    ]

    val_images = [
        p for patient in val_patients
        for p in patients[patient]
    ]

    test_images = [
        p for patient in test_patients
        for p in patients[patient]
    ]

    return train_images, val_images, test_images