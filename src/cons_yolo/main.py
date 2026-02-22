import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO
from pathlib import Path
from collections import Counter

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Cache model loading - models are loaded only once per type
@st.cache_resource
def load_model():
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / "best_construction.pt"
    return YOLO(model_path)

# Cache annotators - reuse same annotator objects
@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    return sv.BoxAnnotator(), sv.LabelAnnotator()

def detector_pipeline_pillow(image_bytes, model):
    """Optimized detection pipeline"""
    # Load and convert image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)
    
    # Run inference
    results = model(image_np_rgb, verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results)
    
    # Get cached annotators
    box_annotator, label_annotator = get_annotators()
    
    # Annotate image
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image_np = np.asarray(annotated_image)
    
    # Optimized class counting using Counter
    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))
    
    return annotated_image_np, classcounts


# --- Bagian Streamlit Utama ---


st.title("🎯 Object Detection")

with st.spinner(f"Loading model..."):
    model = load_model()

st.success(f"✅ model loaded!")

uploaded_file = st.file_uploader(
    "Upload Image",
    accept_multiple_files=False,
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Objects", type="primary"):
        bytes_data = uploaded_file.getvalue()

        with st.spinner("Detecting objects..."):
            annotated_image_rgb, classcounts = detector_pipeline_pillow(bytes_data, model)

        st.subheader("Detection Results")
        st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)

        if classcounts:
            st.subheader("📊 Object Counts")
            col1, col2 = st.columns([1, 2])
            with col1:
                for class_name, count in classcounts.items():
                    st.metric(label=class_name, value=count)

            st.write(classcounts)

            # =========================
            # Perbaikan logic warning
            # =========================
            persons = int(classcounts.get("person", 0))
            helmets = int(classcounts.get("helmet", 0))
            vests = int(classcounts.get("vest", 0))
            no_helmets = int(classcounts.get("no helmet", 0))
            no_vests = int(classcounts.get("no vest", 0))

            if persons > 0:
                st.warning(f"👤 Person detected: {persons}")

                # Prioritas pakai kelas 'no helmet' / 'no vest' jika tersedia
                # Jika tidak ada, fallback pakai selisih person - item
                missing_helmet = no_helmets if no_helmets > 0 else max(persons - helmets, 0)
                missing_vest   = no_vests   if no_vests   > 0 else max(persons - vests, 0)

                # Warning terpisah
                if missing_vest > 0:
                    st.error(f"🦺 No vest detected: {missing_vest} person")

                if missing_helmet > 0:
                    st.error(f"⛑️ No helmet detected: {missing_helmet} person")

                # Warning "tidak pakai keduanya"
                # Kalau model memang bisa mendeteksi kedua pelanggaran pada orang yang sama,
                # pendekatan praktis adalah ambil minimum (perkiraan jumlah yang kena dua-duanya).
                both_missing = min(missing_helmet, missing_vest)
                if both_missing > 0:
                    st.error(f"⚠️ No helmet AND no vest: {both_missing} person")
        else:
            st.info("No objects detected in the image.")