import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import Counter

# =========================================================
# PATH MODEL
# =========================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# =========================================================
# CACHE: LOAD MODEL (sekali saja)
# =========================================================
@st.cache_resource
def load_model():
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / "best_construction.pt"
    return YOLO(model_path)

# =========================================================
# CACHE: ANNOTATORS (sekali saja)
# =========================================================
@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    return sv.BoxAnnotator(), sv.LabelAnnotator()

# =========================================================
# PIPELINE DETEKSI + CONFIDENCE THRESHOLD (AKTIF)
# =========================================================
def detector_pipeline_pillow(image_bytes, model, conf_threshold: float = 0.45):
    """Optimized detection pipeline + confidence filtering"""
    # Load and convert image
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np_rgb = np.array(pil_image)

    # Run inference (aktifkan threshold via parameter YOLO)
    results = model(image_np_rgb, verbose=False, conf=conf_threshold)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter ulang (opsional, kalau confidence tersedia di detections)
    if hasattr(detections, "confidence") and detections.confidence is not None:
        detections = detections[detections.confidence >= conf_threshold]

    # Get cached annotators
    box_annotator, label_annotator = get_annotators()

    # Annotate image
    annotated_image = pil_image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image_np = np.asarray(annotated_image)

    # Class counting
    class_names = detections.data.get("class_name", [])
    classcounts = dict(Counter(class_names))

    return annotated_image_np, classcounts


# =========================================================
# STREAMLIT APP
# =========================================================

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="PPE Detection Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    /* Mempercantik Card & Container */
    [data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Mempercantik Tombol Utama */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
        border: none;
        height: 3em;
    }
    /* Dropzone Upload */
    section[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #ff4b4b;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. GET MODEL (CACHE WRAPPER)
# ==========================================
@st.cache_resource
def get_model():
    return load_model()

# ==========================================
# 4. SIDEBAR SETTINGS (AKTIF)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Settings")
    st.subheader("Model Configuration")

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    st.session_state["conf_threshold"] = conf_threshold  # <-- disimpan untuk dipakai saat inferensi

    st.divider()
    with st.status("Model Status..."):
        model = get_model()
        st.write("✅ Model Loaded")

    st.info("Sistem ini mendeteksi Person, Helmet, Vest, dan pelanggaran APD.")

# ==========================================
# 5. HEADER UTAMA
# ==========================================
st.title("🎯 Object Detection Dashboard")
st.write("Sistem Pemantauan Alat Pelindung Diri (APD) Otomatis")

# ==========================================
# 6. LAYOUT UTAMA (Dua Kolom)
# ==========================================
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📤 Input Section")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Seret dan lepas file gambar di sini"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original Image", use_container_width=True)

        if st.button("🔍 Run PPE Detection"):
            with st.spinner("Analyzing Safety Equipment..."):
                bytes_data = uploaded_file.getvalue()

                # Ambil threshold dari sidebar
                conf_threshold = st.session_state.get("conf_threshold", 0.45)

                # Jalankan deteksi (threshold AKTIF)
                annotated_image, classcounts = detector_pipeline_pillow(
                    bytes_data, model, conf_threshold=conf_threshold
                )

                # Simpan hasil ke session_state
                st.session_state['result_img'] = annotated_image
                st.session_state['counts'] = classcounts
                st.session_state['detection_run'] = True

            st.toast("Detection Completed!", icon="✅")

with col_result:
    st.subheader("🖼️ Detection Result")

    if st.session_state.get('detection_run'):
        # 1. Tampilkan Gambar Ter-annotasi
        st.image(st.session_state['result_img'], caption="Detected Objects", use_container_width=True)

        # 2. Tampilkan Metric Ringkas
        counts = st.session_state['counts']
        if counts:
            m1, m2, m3 = st.columns(3)
            m1.metric("Person", counts.get("person", 0))
            m2.metric("Helmet", counts.get("helmet", 0))
            m3.metric("Vest", counts.get("vest", 0))

            # 3. Logika Warning & Pelanggaran
            st.markdown("---")
            st.subheader("📊 Safety Analysis")

            persons = int(counts.get("person", 0))
            helmets = int(counts.get("helmet", 0))
            vests = int(counts.get("vest", 0))
            no_helmets = int(counts.get("no helmet", 0))
            no_vests = int(counts.get("no vest", 0))

            if persons > 0:
                missing_helmet = no_helmets if no_helmets > 0 else max(persons - helmets, 0)
                missing_vest = no_vests if no_vests > 0 else max(persons - vests, 0)
                both_missing = min(missing_helmet, missing_vest)

                if missing_vest > 0:
                    st.error(f"🦺 **No Vest:** {missing_vest} person detected")

                if missing_helmet > 0:
                    st.error(f"⛑️ **No Helmet:** {missing_helmet} person detected")

                if both_missing > 0:
                    st.warning(f"⚠️ **Major Violation:** {both_missing} person ignoring both APD")

                if missing_helmet == 0 and missing_vest == 0:
                    st.success("✅ All persons are wearing proper PPE.")
        else:
            st.info("No objects detected.")
    else:
        st.info("Hasil deteksi akan muncul di sini setelah proses analisis.")

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()
st.caption("Developed with ❤️ using Streamlit, Supervision, and YOLOv8")