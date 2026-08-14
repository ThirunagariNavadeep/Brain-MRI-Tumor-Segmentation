import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model import UNet
from src.preprocessing import preprocess_image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/best_unet.pth"


@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


def predict(image, model):
    image_array = np.array(image.convert("L"))

    image_array = cv2.resize(
        image_array,
        (256, 256)
    )

    processed = preprocess_image(image_array)

    tensor = torch.from_numpy(
        processed
    ).float().unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        prediction = torch.sigmoid(
            model(tensor)
        )

    mask = (
        prediction[0, 0].cpu().numpy() >= 0.3
    ).astype(np.uint8)

    return image_array, processed, mask


st.set_page_config(
    page_title="Brain MRI Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain MRI Tumor Segmentation")

st.write(
    "Computer vision pipeline for detecting and segmenting "
    "tumor regions in brain MRI slices using OpenCV and U-Net."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI slice",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    model = load_model()

    original, processed, mask = predict(
        image,
        model
    )

    tumor_pixels = int(mask.sum())
    total_pixels = mask.size

    tumor_percentage = (
        tumor_pixels / total_pixels
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original MRI")
        st.image(
            original,
            use_container_width=True
        )

    with col2:
        st.subheader("Predicted Tumor Mask")
        st.image(
            mask * 255,
            use_container_width=True
        )

    with col3:
        st.subheader("Tumor Overlay")

        overlay = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB
        )

        overlay[mask == 1] = [255, 0, 0]

        blended = cv2.addWeighted(
            cv2.cvtColor(
                original,
                cv2.COLOR_GRAY2RGB
            ),
            0.7,
            overlay,
            0.3,
            0
        )

        st.image(
            blended,
            use_container_width=True
        )

    st.divider()

    st.subheader("Quantitative Analysis")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Tumor Pixels",
            f"{tumor_pixels:,}"
        )

    with metric2:
        st.metric(
            "Tumor Area",
            f"{tumor_percentage:.2f}%"
        )

    st.caption(
        "Tumor area represents the percentage of pixels classified "
        "as tumor in the processed 256×256 MRI slice."
    )