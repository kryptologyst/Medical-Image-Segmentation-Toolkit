"""Streamlit demo application for medical image segmentation."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

from src.models.unet import UNet, UNetPlusPlus
from src.utils.device import get_device
from src.utils.explainability import ExplainabilityAnalyzer
from src.metrics.segmentation_metrics import compute_dice_score, compute_iou_score


# Page configuration
st.set_page_config(
    page_title="Medical Image Segmentation Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Medical Image Segmentation Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research demonstration tool and is NOT intended for clinical use.</strong></p>
    <ul>
        <li>This tool is for educational and research purposes only</li>
        <li>Results should NOT be used for medical diagnosis or treatment decisions</li>
        <li>Always consult qualified healthcare professionals for medical advice</li>
        <li>No patient data is stored or transmitted through this application</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Architecture",
    ["UNet", "UNet++"],
    help="Choose the segmentation model architecture"
)

# Load model checkbox
load_pretrained = st.sidebar.checkbox(
    "Load Pretrained Model",
    value=False,
    help="Load a pretrained model checkpoint"
)

# Model file upload
model_file = None
if load_pretrained:
    model_file = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pth', 'pt'],
        help="Upload a trained model checkpoint file"
    )

# Processing options
st.sidebar.subheader("Processing Options")
threshold = st.sidebar.slider(
    "Segmentation Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Threshold for binary segmentation"
)

enable_explainability = st.sidebar.checkbox(
    "Enable Explainability Analysis",
    value=True,
    help="Generate Grad-CAM and uncertainty maps"
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Medical Image")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a medical image file",
        type=['png', 'jpg', 'jpeg', 'nii', 'nii.gz', 'dcm'],
        help="Upload a medical image for segmentation"
    )
    
    # Display uploaded image
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load and display image
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(uploaded_file)
                image_array = np.array(image.convert('L'))  # Convert to grayscale
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to tensor format
                image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
                image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
                
            else:
                st.error("Currently only PNG, JPG, and JPEG formats are supported in the demo.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
    else:
        st.info("Please upload a medical image to begin segmentation.")
        st.stop()

with col2:
    st.subheader("🔬 Segmentation Results")
    
    if uploaded_file is not None:
        # Initialize model
        @st.cache_resource
        def load_model():
            if model_type == "UNet":
                model = UNet(in_channels=1, out_channels=1)
            else:  # UNet++
                model = UNetPlusPlus(in_channels=1, out_channels=1)
            
            # Load pretrained weights if available
            if load_pretrained and model_file is not None:
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    st.success("✅ Pretrained model loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Could not load pretrained model: {e}")
                    st.info("Using randomly initialized model.")
            
            model.eval()
            return model
        
        model = load_model()
        
        # Perform segmentation
        with st.spinner("Performing segmentation..."):
            try:
                with torch.no_grad():
                    # Move to device
                    device = get_device()
                    model = model.to(device)
                    image_tensor = image_tensor.to(device)
                    
                    # Forward pass
                    prediction = model(image_tensor)
                    prediction_prob = torch.sigmoid(prediction)
                    
                    # Convert to numpy
                    prediction_np = prediction_prob.squeeze().cpu().numpy()
                    binary_mask = (prediction_np > threshold).astype(np.uint8)
                    
                    # Display results
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Probability map
                    im1 = axes[0].imshow(prediction_np, cmap='hot')
                    axes[0].set_title('Segmentation Probability')
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0])
                    
                    # Binary mask
                    axes[1].imshow(binary_mask, cmap='gray')
                    axes[1].set_title(f'Binary Mask (threshold={threshold})')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Compute metrics (if we had ground truth)
                    st.subheader("📊 Segmentation Metrics")
                    
                    # Display probability statistics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric(
                            "Max Probability",
                            f"{prediction_np.max():.3f}"
                        )
                    
                    with col_metric2:
                        st.metric(
                            "Mean Probability",
                            f"{prediction_np.mean():.3f}"
                        )
                    
                    with col_metric3:
                        st.metric(
                            "Segmented Area (%)",
                            f"{(binary_mask.sum() / binary_mask.size) * 100:.1f}%"
                        )
                    
                    # Explainability analysis
                    if enable_explainability:
                        st.subheader("🔍 Explainability Analysis")
                        
                        with st.spinner("Generating explainability maps..."):
                            try:
                                analyzer = ExplainabilityAnalyzer(model)
                                explain_results = analyzer.analyze(image_tensor)
                                
                                if explain_results.get("gradcam") is not None:
                                    # Display Grad-CAM
                                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                                    
                                    # Original image
                                    axes[0].imshow(image_array, cmap='gray')
                                    axes[0].set_title('Original Image')
                                    axes[0].axis('off')
                                    
                                    # Grad-CAM overlay
                                    gradcam = explain_results["gradcam"]
                                    axes[1].imshow(image_array, cmap='gray', alpha=0.7)
                                    im = axes[1].imshow(gradcam, cmap='jet', alpha=0.3)
                                    axes[1].set_title('Grad-CAM Overlay')
                                    axes[1].axis('off')
                                    plt.colorbar(im, ax=axes[1])
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                if explain_results.get("uncertainty") is not None:
                                    # Display uncertainty
                                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                                    
                                    # Prediction
                                    axes[0].imshow(prediction_np, cmap='hot')
                                    axes[0].set_title('Prediction')
                                    axes[0].axis('off')
                                    
                                    # Uncertainty
                                    uncertainty = explain_results["uncertainty"]
                                    im = axes[1].imshow(uncertainty, cmap='viridis')
                                    axes[1].set_title('Uncertainty Map')
                                    axes[1].axis('off')
                                    plt.colorbar(im, ax=axes[1])
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Uncertainty statistics
                                    st.metric(
                                        "Mean Uncertainty",
                                        f"{uncertainty.mean():.3f}"
                                    )
                                
                            except Exception as e:
                                st.warning(f"Explainability analysis failed: {e}")
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    # Create downloadable image
                    result_image = Image.fromarray((prediction_np * 255).astype(np.uint8))
                    
                    # Convert to bytes
                    img_buffer = io.BytesIO()
                    result_image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Segmentation Result",
                        data=img_bytes,
                        file_name="segmentation_result.png",
                        mime="image/png"
                    )
                    
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                st.info("Please try with a different image or check the model configuration.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Medical Image Segmentation Demo | Research Tool Only | Not for Clinical Use</p>
    <p>Built with PyTorch, Streamlit, and MONAI</p>
</div>
""", unsafe_allow_html=True)
