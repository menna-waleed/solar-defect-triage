
import streamlit as st
import numpy as np
import joblib

from PIL import Image
from skimage.feature import hog

from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Solar Defect Triage Assistant",
    page_icon="☀️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "solar_svm_model.joblib"

IMG_SIZE = (96, 96)

CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# Extract HOG features
# Must match training settings
# -----------------------------
def extract_hog(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)

    gray = np.dot(
        image_array[..., :3],
        [0.299, 0.587, 0.114]
    ).astype(np.uint8)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


# -----------------------------
# App interface
# -----------------------------
st.title("☀️ Solar Defect Triage Assistant")

st.write(
    "Upload a solar panel image to classify it "
    "and identify cases that may need human review."
)

st.info(
    "This prototype uses a trained machine learning "
    "model with HOG image features."
)

uploaded_file = st.file_uploader(
    "Upload a solar panel image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    try:
        model = load_model()

        features = extract_hog(image)

        prediction = model.predict(features)[0]

        with col2:
            st.subheader("Prediction")
            st.success(f"Predicted class: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]

                order = np.argsort(probabilities)[::-1]

                confidence = float(probabilities[order[0]])

                if len(order) > 1:
                    margin = float(
                        probabilities[order[0]]
                        - probabilities[order[1]]
                    )
                else:
                    margin = 1.0

                st.metric(
                    "Top prediction score",
                    f"{confidence:.1%}"
                )

                st.subheader("Class probabilities")

                for idx in order:
                    label = model.classes_[idx]
                    score = float(probabilities[idx])

                    st.write(f"**{label}** — {score:.1%}")
                    st.progress(score)

                needs_review = (
                    confidence < CONFIDENCE_THRESHOLD
                    or margin < MARGIN_THRESHOLD
                )

                st.subheader("Intelligent Review")

                if needs_review:
                    st.warning(
                        "Human review recommended. "
                        "The prediction score is low or "
                        "the top two classes are too close."
                    )
                else:
                    st.success(
                        "No review flag based on the current "
                        "thresholds. This does not guarantee "
                        "that the prediction is correct."
                    )

                st.caption(
                    "Review thresholds are preliminary and "
                    "should be validated on a separate "
                    "validation dataset."
                )

            else:
                st.warning(
                    "This model does not support probability "
                    "estimates. Review thresholds cannot "
                    "be applied."
                )

    except Exception as e:
        st.error(f"Could not process the image: {e}")
        st.caption(
            "Check that the saved model uses the same HOG "
            "settings and input feature dimensions."
        )
