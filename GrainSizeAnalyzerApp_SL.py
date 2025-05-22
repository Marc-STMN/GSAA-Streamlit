import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from cellpose import models
from skimage import measure
import pandas as pd
import tempfile
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# App Config (must be first Streamlit command)
# --------------------------------------------------
st.set_page_config(page_title="Grain Size Analyzer", layout="wide")

# --------------------------------------------------
# Model Loading with Caching for Performance
# --------------------------------------------------
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)
    cp_model = models.CellposeModel(model_type='cyto', gpu=False)
    return reader, cp_model

reader, cp_model = load_models()

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("Grain Size Analyzer")

# --------------------------------------------------
# Image Upload
# --------------------------------------------------
uploaded = st.file_uploader("Upload SEM Image", type=["jpg", "png", "tif", "tiff"])

if uploaded:
    # Read image bytes into OpenCV
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image. Please upload a valid image file.")
        st.stop()

    # Convert to grayscale and prepare PIL image
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    st.image(pil_img, caption="Kontrolle: SEM-Bild", use_container_width=True)

    st.subheader("Manuelle ROI-Eingabe für Scale-Bar")
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("x (links)", min_value=0, max_value=pil_img.width-1, value=0)
        y = st.number_input("y (oben)", min_value=0, max_value=pil_img.height-1, value=0)
    with col2:
        w = st.number_input("Breite (w)", min_value=1, max_value=pil_img.width-x, value=100)
        h = st.number_input("Höhe (h)", min_value=1, max_value=pil_img.height-y, value=20)

    if st.button("ROI anzeigen"):
        img_annot = img_rgb.copy()
        cv2.rectangle(img_annot, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.image(img_annot, caption="Bild mit ROI", use_container_width=True)

        # Jetzt kannst du wie gewohnt mit gray[y:y+h, x:x+w] weiterarbeiten

        # --------------------------------------------------
        # Capture H-bar Template
        # --------------------------------------------------
        if st.button("Capture Template"):
            roi_gray = gray[y : y + h, x : x + w]
            st.info("Template captured. Proceed to scale extraction.")

        # --------------------------------------------------
        # Scale Extraction
        # --------------------------------------------------
        if st.button("Extract Scale"):
            val_nm = st.number_input("Manually enter scale length (nm)", min_value=0.0, format="%.2f")
            if w == 0:
                st.error("Width of ROI is zero. Please select a valid ROI.")
            else:
                um_per_px = (val_nm / w) / 1000.0
                st.session_state.um_per_px = um_per_px
                st.success(f"Scale set: {val_nm} nm over {w}px = {um_per_px:.4f} µm/px")

        # --------------------------------------------------
        # Run Analysis
        # --------------------------------------------------
        if st.button("Run Analysis"):
            if "um_per_px" not in st.session_state:
                st.error("Please extract scale before running analysis.")
            else:
                h_full, w_full = gray.shape
                crop = gray[: int(0.9 * h_full), :]
                masks, flows, styles = cp_model.eval(crop, diameter=None, channels=[0, 0])
                props = measure.regionprops(masks)
                diams_um = [p.equivalent_diameter * st.session_state.um_per_px for p in props]

                stats = {
                    "mean": np.mean(diams_um),
                    "std": np.std(diams_um),
                    "min": np.min(diams_um),
                    "max": np.max(diams_um),
                    "count": len(diams_um),
                }
                st.subheader("Results")
                st.write(stats)

                # Annotate and display image
                annot = img_bgr.copy()
                for p in props:
                    mask_lbl = (masks == p.label).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(
                        mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annot[: crop.shape[0]], cnts, -1, (0, 0, 255), 1)
                st.image(
                    cv2.cvtColor(annot, cv2.COLOR_BGR2RGB),
                    caption="Segments Annotated",
                    use_container_width=True,
                )

                # Show histogram
                st.subheader("Size Distribution")
                df = pd.DataFrame({"diameter_um": diams_um})
                st.bar_chart(df["diameter_um"].value_counts(bins=20).sort_index())

                # Download annotated image
                buffered = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                cv2.imwrite(
                    buffered.name, cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)
                )
                st.download_button(
                    "Download Annotated Image", buffered.name
                )
