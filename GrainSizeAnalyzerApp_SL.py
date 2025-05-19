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
    reader = easyocr.Reader(['en'], gpu=True)
    cp_model = models.CellposeModel(model_type='cyto', gpu=True)
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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Display original
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --------------------------------------------------
    # ROI Selection for Scale Bar via Canvas
    # --------------------------------------------------
    st.subheader("Select Scale Bar Region")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
        height=img_bgr.shape[0],
        width=img_bgr.shape[1],
        drawing_mode="rect",
        key="canvas",
    )
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        obj = canvas_result.json_data["objects"][0]
        x, y = int(obj["left"]), int(obj["top"])
        w, h = int(obj["width"]), int(obj["height"])
        st.success(f"Scale ROI: x={x}, y={y}, w={w}, h={h}")

        # --------------------------------------------------
        # Capture H-bar Template
        # --------------------------------------------------
        if st.button("Capture Template"):
            roi_gray = gray[y:y+h, x:x+w]
            # (Insert _measure_hbar_width and template cropping here)
            st.info("Template captured. Proceed to scale extraction.")

        # --------------------------------------------------
        # Scale Extraction
        # --------------------------------------------------
        if st.button("Extract Scale"):
            # Perform OCR on cropped bottom-left
            # (Insert OCR preprocessing and extraction logic)
            val_nm = st.number_input("Manually enter scale length", min_value=0.0, format="%.2f")
            w_px = w
            um_per_px = (val_nm/w_px)/1000.0
            st.success(f"Scale set: {val_nm} nm over {w_px}px")

        # --------------------------------------------------
        # Run Analysis
        # --------------------------------------------------
        if st.button("Run Analysis"):
            # Crop top 90% for segmentation
            h_full, w_full = gray.shape
            crop = gray[:int(0.9*h_full), :]
            masks, flows, styles = cp_model.eval(crop, diameter=None, channels=[0,0])
            props = measure.regionprops(masks)
            diams_um = [p.equivalent_diameter*um_per_px for p in props]
            stats = {
                'mean': np.mean(diams_um),
                'std': np.std(diams_um),
                'min': np.min(diams_um),
                'max': np.max(diams_um),
                'count': len(diams_um)
            }
            st.subheader("Results")
            st.write(stats)

            # Display annotated image
            annot = img_bgr.copy()
            for p in props:
                mask_lbl = (masks==p.label).astype(np.uint8)*255
                cnts, _ = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annot[:crop.shape[0]], cnts, -1, (0,0,255), 1)
            st.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), caption="Segments Annotated", use_column_width=True)

            # Display histogram
            st.subheader("Size Distribution")
            df = pd.DataFrame({'diameter_um': diams_um})
            st.bar_chart(df['diameter_um'].value_counts(bins=20).sort_index())

            # Offer download
            buffered = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(buffered.name, cv2.cvtColor(annot, cv2.COLOR_BGR2RGB))
            st.download_button("Download Annotated Image", buffered.name)
