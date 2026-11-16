"""
Historic Photo Restoration Studio
=================================
A modern, interactive Web UI for historical photograph restoration.
Designed for live demonstrations, instant image ingestion (upload & presets),
and immediate side-by-side display of restored photographs.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

# Prevent Matplotlib cache delays in sandbox
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

# Setup project root path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.inference.pipeline import HistoricImageRestorationPipeline, RestorationConfig

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Historic Photo Restoration Studio",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, premium modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: #f8fafc;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .badge-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .badge-success {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    .badge-accent {
        background-color: #312e81;
        color: #a5b4fc;
        border: 1px solid #4338ca;
    }
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .ingest-box {
        background-color: #0f172a;
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .demo-card {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px;
        background-color: #1e293b;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Cache the restoration engine to load models once."""
    return HistoricImageRestorationPipeline()

pipeline = get_pipeline()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🎞️ Historic Photo Restoration Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI & Computer Vision engine to repair physical tears, scratches, chemical fading, film grain, and optical softness in archival photographs.</div>',
        unsafe_allow_html=True
    )

device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
with col_status:
    st.markdown(f"""
    <div class="badge-bar" style="justify-content: flex-end; margin-top: 0.8rem;">
        <span class="badge badge-success">● Engine Live</span>
        <span class="badge badge-accent">Demo Mode Active</span>
        <span class="badge">{device_str}</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Parameter Controls & Studio Settings
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ Restoration Controls")

# Preset Profiles
profile_choice = st.sidebar.selectbox(
    "Restoration Preset Profile",
    options=[
        "🌟 Balanced Restoration (Recommended)",
        "🩹 Physical Damage & Tears Focus",
        "🎨 Color, Fading & Contrast Revival",
        "🔍 Sharpness & Clarity Enhancement",
        "⚙️ Custom Studio Mode",
    ],
    index=0,
)

# Set defaults based on profile
if "Balanced" in profile_choice:
    def_sens, def_dilate, def_inpaint_m, def_inpaint_p = 60, 2, "telea", 1
    def_clahe, def_gray, def_denoise, def_sharpen = 1.5, 0.3, 4, 0.9
    def_dark = False
elif "Physical Damage" in profile_choice:
    def_sens, def_dilate, def_inpaint_m, def_inpaint_p = 70, 2, "telea", 1
    def_clahe, def_gray, def_denoise, def_sharpen = 1.2, 0.15, 3, 0.6
    def_dark = False
elif "Color, Fading" in profile_choice:
    def_sens, def_dilate, def_inpaint_m, def_inpaint_p = 50, 1, "telea", 1
    def_clahe, def_gray, def_denoise, def_sharpen = 3.0, 0.65, 4, 0.8
    def_dark = False
elif "Sharpness" in profile_choice:
    def_sens, def_dilate, def_inpaint_m, def_inpaint_p = 50, 1, "telea", 1
    def_clahe, def_gray, def_denoise, def_sharpen = 1.5, 0.2, 2, 1.8
    def_dark = False
else:  # Custom
    def_sens, def_dilate, def_inpaint_m, def_inpaint_p = 60, 2, "telea", 1
    def_clahe, def_gray, def_denoise, def_sharpen = 1.5, 0.3, 4, 0.9
    def_dark = False

# Studio Controls
with st.sidebar.expander("1. Defect & Scratch Detection", expanded=("Custom" in profile_choice or "Damage" in profile_choice)):
    c_bright = st.checkbox("Detect Bright Scratches & Tears", value=True)
    c_dark = st.checkbox("Detect Dark Cracks & Creases", value=def_dark,
                         help="⚠️ Enable only for genuinely dark-cracked images. Causes false positives on textured dark photos.")
    c_sens = st.slider("Detection Sensitivity", min_value=1, max_value=90, value=def_sens, help="Higher = more pixels flagged as defects. Keep at 60-70 for most photos. Only raise if tears are missed.")
    c_dilate = st.slider("Boundary Dilation (px)", min_value=0, max_value=6, value=def_dilate, help="Expands the defect mask to cleanly cover scratch edges. 2 is usually optimal.")
    c_ksize = st.select_slider("Morphological Filter Size", options=[3, 5, 7, 9, 11], value=7)

with st.sidebar.expander("2. Defect Inpainting", expanded=("Custom" in profile_choice)):
    c_inpaint_enable = st.checkbox("Enable Defect Inpainting", value=True)
    c_inpaint_method = st.selectbox(
        "Inpainting Algorithm",
        options=["Telea (Fast Marching)", "Navier-Stokes (Fluid dynamics)"],
        index=0 if def_inpaint_m == "telea" else 1,
    )
    inpaint_code = "telea" if "Telea" in c_inpaint_method else "ns"
    c_inpaint_radius = st.slider("Inpainting Neighborhood Radius", min_value=1, max_value=10, value=5)
    c_inpaint_passes = st.slider("Recursive Inpainting Passes", min_value=1, max_value=4, value=def_inpaint_p, help="1 pass is best for most images. Use 2 only for very thick tears.")

with st.sidebar.expander("3. Tone & Color Recovery", expanded=("Custom" in profile_choice or "Color" in profile_choice)):
    c_color_enable = st.checkbox("Enable Tone & Color Recovery", value=True)
    c_clahe = st.slider("CLAHE Dynamic Range Clip", min_value=0.0, max_value=5.0, value=float(def_clahe), step=0.1, help="Recovers shadow and highlight details in aged photos.")
    c_gray = st.slider("Gray-World White Balance", min_value=0.0, max_value=1.0, value=float(def_gray), step=0.05, help="Removes yellow/sepia color casts.")
    c_sat = st.slider("Color Saturation", min_value=0.8, max_value=1.5, value=1.05, step=0.05)

with st.sidebar.expander("4. Denoising & Grain Filter", expanded=("Custom" in profile_choice)):
    c_denoise_enable = st.checkbox("Enable Film Grain Denoising", value=True)
    c_denoise = st.slider("Bilateral Filter Strength", min_value=1, max_value=15, value=def_denoise, help="Smooths film grain while preserving sharp edges.")

with st.sidebar.expander("5. Lens & Detail Sharpening", expanded=("Custom" in profile_choice or "Sharpness" in profile_choice)):
    c_sharpen_enable = st.checkbox("Enable Detail Sharpening", value=True)
    c_sharpen = st.slider("Unsharp Mask Amount", min_value=0.0, max_value=2.5, value=float(def_sharpen), step=0.1)

with st.sidebar.expander("6. Deep Learning Models (Optional)", expanded=False):
    st.markdown("**U-Net Restoration Checkpoint**")
    default_ckpt = str(APP_DIR / "checkpoints" / "best_model.pth")
    c_use_unet = st.checkbox("Use Trained U-Net Model", value=False)
    c_ckpt_path = st.text_input("Checkpoint Path (.pth)", value=default_ckpt if Path(default_ckpt).exists() else "")
    c_unet_passes = st.slider("U-Net Recursive Passes", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("**CodeFormer AI Face Restoration (Cloud)**")
    c_use_codeformer = st.checkbox("Enable CodeFormer AI (via Hugging Face API)", value=False)
    c_hf_token = st.text_input("Hugging Face API Token", value="", type="password", help="Requires internet access to sczhou/CodeFormer space.")
    c_fidelity = st.slider("CodeFormer Fidelity", min_value=0.1, max_value=1.0, value=0.7, step=0.05)

# Assemble config
config = RestorationConfig(
    detect_bright_scratches=c_bright,
    detect_dark_cracks=c_dark,
    scratch_sensitivity=c_sens,
    scratch_kernel_size=c_ksize,
    scratch_dilation_radius=c_dilate,
    enable_inpainting=c_inpaint_enable,
    inpaint_method=inpaint_code,
    inpaint_radius=c_inpaint_radius,
    inpaint_passes=c_inpaint_passes,
    enable_color_correction=c_color_enable,
    clahe_clip_limit=c_clahe,
    gray_world_weight=c_gray,
    saturation_multiplier=c_sat,
    enable_denoising=c_denoise_enable,
    denoise_strength=c_denoise,
    enable_sharpening=c_sharpen_enable,
    sharpen_amount=c_sharpen,
    use_unet=c_use_unet,
    unet_passes=c_unet_passes,
    use_codeformer=c_use_codeformer,
    hf_token=c_hf_token if c_hf_token else None,
    codeformer_fidelity=c_fidelity,
)

if c_use_unet and c_ckpt_path:
    if Path(c_ckpt_path).exists():
        pipeline.load_unet(c_ckpt_path)
    else:
        st.sidebar.warning(f"Checkpoint not found at: {c_ckpt_path}. Local algorithmic restoration will be used.")

# -----------------------------------------------------------------------------
# Image Ingestion Section: Live Upload & 3 Official Demonstration Images
# -----------------------------------------------------------------------------
demo_dir = APP_DIR / "demo_images"

demo_catalog = {
    "🌟 Demo 1: Forest Hiker — Large Vertical Tear": {
        "file": demo_dir / "user_damaged_1.png",
        "clean": demo_dir / "user_clean_1.png",
        "desc": "Grayscale photo of a hiker in dense forest with a prominent white vertical tear running through the centre. The pipeline detects and seamlessly removes the tear while preserving natural texture."
    },
    "🏜️ Demo 2: Desert Dune — Jagged Horizontal Tear": {
        "file": demo_dir / "user_damaged_3.png",
        "clean": demo_dir / "user_clean_3.png",
        "desc": "Sandy desert dune landscape with a wide, jagged bright tear across the horizon. CLAHE contrast recovery and tear inpainting restore the warm-toned scene."
    },
    "🌃 Demo 3: Night Scene — Dark Sky Damage": {
        "file": demo_dir / "user_damaged_2.png",
        "clean": demo_dir / "user_clean_2.png",
        "desc": "Low-light night landscape with dust marks, chemical degradation, and noise. Bilateral denoising and CLAHE dynamic range recovery bring back detail from the dark scene."
    },
    "1️⃣ Archive 1: Vintage Historical Portrait": {
        "file": demo_dir / "demo_1_portrait_damaged.png",
        "clean": demo_dir / "demo_1_portrait_clean_reference.png",
        "desc": "Damaged 20th-century portrait with prominent facial scratches, emulsion tears, and chemical sepia fading."
    },
    "2️⃣ Archive 2: Heritage Monument (Diagonal Fissures)": {
        "file": demo_dir / "demo_2_heritage_monument_damaged.png",
        "clean": demo_dir / "demo_2_heritage_monument_clean_reference.png",
        "desc": "Historic architectural monument photograph with deep diagonal surface fissures and faded shadows."
    },
}

st.markdown("### 📥 Ingest Photograph for Restoration")
ingest_mode = st.radio(
    "Choose Image Ingestion Method:",
    options=[
        "📁 Upload Image from My Computer (Drag & Drop or Browse)",
        "🎯 Select from Curated Demonstration Images (Includes Your Uploaded Photo)",
    ],
    horizontal=True,
)

input_image_bgr: Optional[np.ndarray] = None
input_image_name: str = "photograph"
reference_bgr: Optional[np.ndarray] = None

if "Upload" in ingest_mode:
    uploaded_file = st.file_uploader(
        "Drop or select an image file to ingest into the restoration engine:",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        help="Upload any old, scratched, faded, or blurry photograph.",
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_image_name = Path(uploaded_file.name).stem
        st.success(f"✅ Ingested uploaded image: **{uploaded_file.name}** ({input_image_bgr.shape[1]}x{input_image_bgr.shape[0]} px)")
    else:
        st.info("👆 Please select or drag & drop a damaged photo file above to ingest it.")
else:
    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        selected_demo_key = st.selectbox(
            "Select Demonstration Photograph:",
            options=list(demo_catalog.keys()),
            index=0,
        )
    demo_item = demo_catalog[selected_demo_key]
    with col_info:
        st.markdown(f"*{demo_item['desc']}*")

    demo_path = demo_item["file"]
    if demo_path.exists():
        input_image_bgr = cv2.imread(str(demo_path))
        input_image_name = demo_path.stem
        if demo_item["clean"].exists():
            reference_bgr = cv2.imread(str(demo_item["clean"]))
        st.success(f"✅ Ingested demonstration image: **{selected_demo_key}** ({input_image_bgr.shape[1]}x{input_image_bgr.shape[0]} px)")

# -----------------------------------------------------------------------------
# Restoration Execution & Real-Time Display
# -----------------------------------------------------------------------------
if input_image_bgr is not None:
    h, w = input_image_bgr.shape[:2]
    
    # Process image
    with st.spinner("Restoring photograph with AI & Computer Vision..."):
        result = pipeline.restore(
            input_image_bgr,
            config=config,
            reference_image=reference_bgr,
        )

    restored_bgr = result["restored_bgr"]
    restored_rgb = result["restored_rgb"]
    intermediates = result["intermediates"]
    metrics = result["metrics"]
    defect_mask = intermediates.get("defect_mask", np.zeros((h, w), dtype=np.uint8))

    # -------------------------------------------------------------------------
    # Quantitative Quality Metrics Bar
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Restoration Performance & Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics['processing_time_s']:.3f} s</div>
            <div class="metric-lbl">⚡ Processing Speed</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics['defect_percentage']:.2f}%</div>
            <div class="metric-lbl">🩹 Defects Repaired ({metrics['defect_pixel_count']:,} px)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        gain_sign = "+" if metrics['sharpness_gain_pct'] >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{gain_sign}{metrics['sharpness_gain_pct']:.1f}%</div>
            <div class="metric-lbl">🔍 Sharpness Gain ({metrics['sharpness_restored']:.0f} vs {metrics['sharpness_original']:.0f})</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{metrics['contrast_restored']:.2f}</div>
            <div class="metric-lbl">⚖️ Dynamic Contrast (Orig: {metrics['contrast_original']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)

    if "PSNR" in metrics:
        st.info(f"**Ground Truth Reference Comparison:** PSNR: **{metrics['PSNR']:.2f} dB** | SSIM: **{metrics['SSIM']:.4f}** | MAE: **{metrics['MAE']:.2f}**")

    # -------------------------------------------------------------------------
    # Main Image Visualizer: Side-by-Side & Inspection Tabs
    # -------------------------------------------------------------------------
    st.markdown("### 🖼️ Restored Image Display & Visual Inspection")
    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
        "↔️ Before & After Comparison",
        "🎯 Detected Defect & Scratch Mask Overlay",
        "🪜 Step-by-Step Pipeline Evolution",
        "🔥 Reconstruction Difference Heatmap",
    ])

    with v_tab1:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 1. Ingested Damaged Image")
            st.image(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c_right:
            st.markdown("#### 2. Restored Photograph")
            st.image(restored_rgb, use_container_width=True)

    with v_tab2:
        # Create glowing red/amber defect overlay over original
        overlay = input_image_bgr.copy()
        overlay[defect_mask > 0] = [0, 80, 255]  # Bright coral/orange highlight
        blended = cv2.addWeighted(input_image_bgr, 0.40, overlay, 0.60, 0)
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("#### Binary Scratch & Crack Mask")
            st.image(defect_mask, caption="Identified surface scratches, cracks, and physical tears", use_container_width=True)
        with c_m2:
            st.markdown("#### Defect Overlay on Original")
            st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Detected defects highlighted in orange over the photo", use_container_width=True)

    with v_tab3:
        st.markdown("#### Progressive Pipeline Decomposition")
        p_cols = st.columns(len(intermediates))
        for idx, (stage_name, stage_img) in enumerate(intermediates.items()):
            with p_cols[idx]:
                if len(stage_img.shape) == 2:
                    st.image(stage_img, caption=stage_name.capitalize(), use_container_width=True)
                else:
                    st.image(cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB), caption=stage_name.capitalize(), use_container_width=True)

    with v_tab4:
        diff_bgr = intermediates.get("difference", cv2.absdiff(restored_bgr, input_image_bgr))
        diff_norm = cv2.normalize(diff_bgr, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(cv2.cvtColor(diff_norm, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        st.image(
            cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            caption="Heatmap showing regions of greatest modification (warm colors = highest reconstruction)",
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Export & Download Section
    # -------------------------------------------------------------------------
    st.markdown("### 💾 Export Restored Photograph")
    d1, d2, d3 = st.columns(3)
    
    # Encode restored image to PNG buffer
    _, restored_png = cv2.imencode(".png", restored_bgr)
    _, mask_png = cv2.imencode(".png", defect_mask)
    
    # Create Side-by-side composite
    composite = np.hstack([input_image_bgr, restored_bgr])
    _, comp_png = cv2.imencode(".png", composite)

    with d1:
        st.download_button(
            label="⬇️ Download Restored Photo (PNG)",
            data=restored_png.tobytes(),
            file_name=f"{input_image_name}_restored.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Download Defect Mask (PNG)",
            data=mask_png.tobytes(),
            file_name=f"{input_image_name}_defect_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="⬇️ Download Side-by-Side Composite",
            data=comp_png.tobytes(),
            file_name=f"{input_image_name}_before_after.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    st.info("👆 Please ingest an image above (either upload your own photo or choose one of the 3 demonstration photos) to start!")
