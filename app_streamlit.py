import os
import sys
import streamlit as st
import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from common.model import NIH_CLASSES, load_trained_model
    from common.inference import predict_tta, preprocess, probs_to_report
    from common.gradcam import explain
except Exception as e:
    st.error(f"Error importing project modules: {e}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Chest X-ray AI Diagnosis Assistant",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Dark Theme UI/UX Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 100%);
        color: #f1f5f9;
    }
    .main-title {
        text-align: center;
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.6rem;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-weight: 400;
    }
    .dark-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .dark-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
    }
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #3b82f6;
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        width: 100%;
        padding: 12px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-1px);
    }
    .stFileUploader {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='main-title'>🩻 Chest X-ray AI Diagnosis Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Hybrid Deep Learning (EfficientNetV2 + Swin-Small) &bull; Test-Time Augmentation &bull; Explainable Grad-CAM</div>", unsafe_allow_html=True)

WEIGHTS_PATH = os.path.join("models", "FINAL_BEST_EPOCH5_AUC_0.76148.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"No weights found at {WEIGHTS_PATH}.")
    return load_trained_model(WEIGHTS_PATH, device=torch.device(DEVICE))

if not os.path.exists(WEIGHTS_PATH):
    st.warning(f"⚠️ No trained weights found at `models/FINAL_BEST_EPOCH5_AUC_0.76148.pth`.")
else:
    try:
        model = get_model()
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()

    # Main Layout Split: Upload & Action Panel
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
    col_up1, col_up2 = st.columns([2, 1], gap="large")
    
    with col_up1:
        st.markdown("#### 📂 Upload Radiograph")
        uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    with col_up2:
        st.markdown("#### ⚡ Control Panel")
        st.write("Run deep learning inference and generate visual explanations.")
        analyze_btn = st.button("Analyze X-ray", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            st.stop()

        # Handle analysis trigger via button or session cache
        if analyze_btn:
            with st.spinner("Executing Test-Time Augmentation & generating Grad-CAM explanations..."):
                try:
                    mean_probs, top_idx = predict_tta(model, pil_image, device=DEVICE)
                    report = probs_to_report(mean_probs)

                    input_tensor = preprocess(pil_image).to(DEVICE)
                    input_tensor.requires_grad_(True)
                    model.zero_grad()
                    _, _, overlay_img, arrow_img, hotspot = explain(model, input_tensor, top_idx, pil_image)

                    st.session_state['analyzed'] = True
                    st.session_state['top_idx'] = top_idx
                    st.session_state['mean_probs'] = mean_probs
                    st.session_state['report'] = report
                    st.session_state['overlay_img'] = overlay_img
                    st.session_state['arrow_img'] = arrow_img
                    st.session_state['pil_image'] = pil_image
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

        # Results Section
        if st.session_state.get('analyzed', False):
            st.markdown("---")
            st.markdown("### 📊 Diagnostic Results & Visualizations")

            top_idx = st.session_state['top_idx']
            mean_probs = st.session_state['mean_probs']
            top_class = NIH_CLASSES[top_idx]
            top_conf = float(mean_probs[top_idx]) * 100

            # Metric Card Banner
            st.markdown(f"""
                <div class='metric-box'>
                    <h2 style="margin:0 0 5px 0; color:#ffffff; font-size: 1.8rem;">Primary Finding: {top_class}</h2>
                    <p style="margin:0; font-size:1.1rem; color:#60a5fa; font-weight: 500;">Confidence Score: {top_conf:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # 3-Column Visual Comparison Grid
            st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Model Interpretation & Localization")
            r_col1, r_col2, r_col3 = st.columns(3, gap="medium")
            
            with r_col1:
                st.markdown("**Original X-Ray**")
                st.image(st.session_state['pil_image'], use_container_width=True)
                
            with r_col2:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(st.session_state['overlay_img'], use_container_width=True)
                
            with r_col3:
                st.markdown("**Highlighted Region**")
                st.image(st.session_state['arrow_img'], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Probabilities Breakdown
            st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
            st.markdown("#### 📋 Comprehensive Pathology Probabilities")
            st.write("Detailed confidence distribution across all evaluated thoracic pathologies:")
            
            for item in st.session_state['report']:
                label = item["label"]
                prob = float(item["probability"])
                prob_pct = prob * 100
                st.progress(prob, text=f"**{label}**: {prob_pct:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)