import streamlit as st
import numpy as np
import cv2
import time
import os
from PIL import Image
import io
import base64
import random

# Import our custom modules
from models import (
    get_available_models,
    enhance_image_superres,
    enhance_image_deblur,
    enhance_image_denoise,
    enhance_image_restoration,
    enhance_image_colorization,
    enhance_image_sharpen,
    enhance_image_portrait,
    enhance_image_hdr,
    enhance_image_background,
    load_model
)
from utils import (
    get_image_download_link,
    resize_image_if_needed
)
from components.image_comparison import image_comparison_component
from model_manager import models_management_ui

# Set page title and layout
st.set_page_config(
    page_title="AI Image Enhancer Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 3em !important;
        font-weight: bold;
        margin-bottom: 0;
        color: #4F8BF9;
    }
    .sub-title {
        font-size: 1.5em !important;
        margin-top: 0;
        margin-bottom: 2rem;
        color: #888;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #4F8BF9;
    }
    .feature-desc {
        color: #666;
        margin-bottom: 15px;
    }
    .download-btn {
        display: inline-block;
        padding: 10px 20px;
        background-color: #4F8BF9;
        color: white !important;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
        text-align: center;
    }
    .download-btn:hover {
        background-color: #3a7ad5;
    }
    .stProgress .st-bo {
        background-color: #4F8BF9;
    }
    .enhancement-selector {
        margin-bottom: 2rem;
    }
    .parameter-control {
        margin-top: 1rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        text-align: center;
        color: #888;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'selected_enhancement' not in st.session_state:
    st.session_state.selected_enhancement = 'superres'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'enhancement_params' not in st.session_state:
    st.session_state.enhancement_params = {}
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'enhancement_history' not in st.session_state:
    st.session_state.enhancement_history = []

def apply_enhancement(image, enhancement_type, model_name, params=None):
    """
    Apply the selected enhancement to the image.
    """
    if params is None:
        params = {}
    
    # Load the model
    model_config = load_model(model_name, enhancement_type)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"Enhancing image... {int(progress * 100)}%")
    
    # Apply the enhancement based on the type
    start_time = time.time()
    
    try:
        if enhancement_type == 'superres':
            result = enhance_image_superres(image, model_config, params, update_progress)
        elif enhancement_type == 'deblur':
            result = enhance_image_deblur(image, model_config, params, update_progress)
        elif enhancement_type == 'denoise':
            result = enhance_image_denoise(image, model_config, params, update_progress)
        elif enhancement_type == 'restoration':
            result = enhance_image_restoration(image, model_config, params, update_progress)
        elif enhancement_type == 'colorization':
            result = enhance_image_colorization(image, model_config, params, update_progress)
        elif enhancement_type == 'sharpen':
            result = enhance_image_sharpen(image, model_config, params, update_progress)
        elif enhancement_type == 'portrait':
            result = enhance_image_portrait(image, model_config, params, update_progress)
        elif enhancement_type == 'hdr':
            result = enhance_image_hdr(image, model_config, params, update_progress)
        elif enhancement_type == 'background':
            result = enhance_image_background(image, model_config, params, update_progress)
        else:
            st.error(f"Unknown enhancement type: {enhancement_type}")
            return None
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return None
    finally:
        # Ensure progress is complete
        progress_bar.progress(1.0)
        
    # Display processing time
    end_time = time.time()
    status_text.text(f"Enhancement completed in {end_time - start_time:.2f} seconds")
    
    return result

def get_demo_image():
    """Get a demo image for users to try"""
    # In a real app, this would load an actual demo image
    # Here we'll create a simple gradient image for demo purposes
    try:
        # First try to load an actual image from the demo folder if it exists
        demo_folder = "demo"
        if os.path.exists(demo_folder):
            demo_files = [f for f in os.listdir(demo_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if demo_files:
                demo_path = os.path.join(demo_folder, random.choice(demo_files))
                return Image.open(demo_path)
    except:
        pass
    
    # If no demo images, create one
    width, height = 800, 600
    # Create a simple gradient image using NumPy
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels for a colorful gradient
    r = (np.sin(X * 5) + 1) / 2 * 255
    g = (np.cos(Y * 5) + 1) / 2 * 255
    b = (np.sin(X * 2) * np.cos(Y * 2) + 1) / 2 * 255
    
    # Combine channels and convert to image
    img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(img_array)

def image_enhancer_ui():
    """UI for the image enhancer functionality"""
    st.markdown('<h1 class="main-title">AI Image Enhancer Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enhance your photos with state-of-the-art AI models</p>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.title("Enhancement Controls")
    
    # Load available models
    available_models = get_available_models()
    
    # Enhancement type selection
    enhancement_options = []
    for key, value in available_models.items():
        enhancement_options.append({"id": key, "name": value["display_name"], "description": value["description"]})
    
    # Sort options by display name
    enhancement_options.sort(key=lambda x: x["name"])
    
    # Dropdown for enhancement selection with descriptions
    st.sidebar.markdown("### Select Enhancement Type")
    enhancement_type = st.sidebar.selectbox(
        "Enhancement Type",
        options=[option["id"] for option in enhancement_options],
        format_func=lambda x: next((option["name"] for option in enhancement_options if option["id"] == x), x),
        index=0
    )
    
    # Get the selected enhancement description
    selected_enhancement_desc = next((option["description"] for option in enhancement_options if option["id"] == enhancement_type), "")
    st.sidebar.markdown(f"*{selected_enhancement_desc}*")
    
    # Model selection for the chosen enhancement type
    st.sidebar.markdown("### Select Model")
    available_models_for_type = available_models[enhancement_type]["models"]
    model_options = list(available_models_for_type.items())
    
    selected_model = st.sidebar.selectbox(
        "Model",
        options=[model[0] for model in model_options],
        format_func=lambda x: next((model[1] for model in model_options if model[0] == x), x),
        index=0
    )
    
    # Parameters for the enhancement based on the type
    st.sidebar.markdown("### Enhancement Parameters")
    
    # Different parameters based on enhancement type
    enhancement_params = {}
    
    if enhancement_type == 'superres':
        enhancement_params['scale'] = st.sidebar.slider("Scale Factor", 1.0, 4.0, 2.0, 0.5)
    elif enhancement_type == 'denoise':
        enhancement_params['strength'] = st.sidebar.slider("Denoising Strength", 5, 30, 15, 1)
    elif enhancement_type == 'sharpen':
        enhancement_params['strength'] = st.sidebar.slider("Sharpening Strength", 1.0, 3.0, 1.5, 0.1)
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image to enhance", type=["jpg", "jpeg", "png"])
    
    # Demo button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Try with Demo Image"):
            demo_image = get_demo_image()
            st.session_state.original_image = demo_image
            st.session_state.current_image = demo_image
            st.session_state.enhanced_image = None  # Reset enhanced image
            st.session_state.enhancement_history = []
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            
            # Resize if the image is too large
            pil_image = resize_image_if_needed(pil_image)
            
            # Update session state
            st.session_state.original_image = pil_image
            st.session_state.current_image = pil_image
            st.session_state.enhanced_image = None  # Reset enhanced image
            st.session_state.enhancement_history = []
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
    
    # Enhance button
    if st.session_state.original_image is not None:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Enhance Image"):
                # Apply the selected enhancement
                with st.spinner("Enhancing image..."):
                    # Use current image, which may have already been enhanced
                    source_image = st.session_state.current_image or st.session_state.original_image
                    
                    enhanced = apply_enhancement(
                        source_image,
                        enhancement_type,
                        selected_model,
                        enhancement_params
                    )
                    
                    if enhanced is not None:
                        # Add to enhancement history
                        enhancement_name = next((option["name"] for option in enhancement_options if option["id"] == enhancement_type), "")
                        st.session_state.enhancement_history.append({
                            "type": enhancement_type,
                            "name": enhancement_name,
                            "model": selected_model,
                            "params": enhancement_params
                        })
                        
                        # Update the enhanced image and current image
                        st.session_state.enhanced_image = enhanced
                        st.session_state.current_image = enhanced
                        st.session_state.selected_enhancement = enhancement_type
                        st.session_state.selected_model = selected_model
                        st.session_state.enhancement_params = enhancement_params
        
        # Display current working image
        st.write("### Current Image")
        st.image(st.session_state.current_image, use_container_width=True)
        
        # If we have an enhancement history, show it
        if st.session_state.enhancement_history:
            st.write("### Enhancement History")
            
            for i, enhancement in enumerate(st.session_state.enhancement_history):
                st.write(f"**{i+1}. {enhancement['name']}** (Model: {enhancement['model']})")
                # Format parameters for display
                params_str = ", ".join([f"{k}: {v}" for k, v in enhancement['params'].items()])
                if params_str:
                    st.write(f"   Parameters: {params_str}")
            
            # Download button
            st.markdown(
                get_image_download_link(st.session_state.current_image, "enhanced_image.png", "Download Enhanced Image"),
                unsafe_allow_html=True
            )
            
            # Display side-by-side comparison
            st.write("### Before & After Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original**")
                st.image(st.session_state.original_image, use_container_width=True)
            with col2:
                st.write("**Current Enhanced**")
                st.image(st.session_state.current_image, use_container_width=True)
            
            # Option to reset to original
            if st.button("Reset to Original Image"):
                st.session_state.current_image = st.session_state.original_image
                st.session_state.enhanced_image = None
                st.session_state.enhancement_history = []
                st.rerun()
    
    # Display features
    if st.session_state.original_image is None:
        st.write("## Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔍 Super Resolution</div>
                <div class="feature-desc">Increase the resolution of your images while preserving details.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">🎨 Colorization</div>
                <div class="feature-desc">Add vibrant colors to black and white photographs.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">✨ Restoration</div>
                <div class="feature-desc">Restore old or damaged photos to their former glory.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">👥 Portrait Enhancement</div>
                <div class="feature-desc">Improve facial features and skin tones in portraits.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔆 HDR Effect</div>
                <div class="feature-desc">Add dynamic range to flat images for a more vivid look.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">💫 Deblurring</div>
                <div class="feature-desc">Fix blurry photos with advanced AI techniques.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">🧹 Denoising</div>
                <div class="feature-desc">Remove noise from images taken in low-light conditions.</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-title">🖼️ Background Enhancement</div>
                <div class="feature-desc">Enhance or blur backgrounds while preserving the subject.</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        AI Image Enhancer Pro © 2025 | Built with Streamlit and PyTorch
    </div>
    """, unsafe_allow_html=True)

# Main app with tabs
def main():
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Image Enhancer", "Models Management"])
    
    with tab1:
        image_enhancer_ui()
    
    with tab2:
        models_management_ui()

if __name__ == "__main__":
    main()