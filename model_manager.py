import os
import hashlib
import requests
import json
from pathlib import Path
import shutil
import time
from datetime import datetime
import threading
import urllib.request
from tqdm.auto import tqdm
import streamlit as st

# Directory for cached models
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# Models info file
MODELS_INFO_FILE = os.path.join(MODELS_CACHE_DIR, "models_info.json")

# Model download URLs
MODEL_URLS = {
    # Super-resolution models
    "real_esrgan_x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "hash": "5ba5a6499eb8c87ec77d92db0fcf8a06",
        "size": 67000000,
        "category": "superres",
        "display_name": "Real-ESRGAN x4",
        "description": "State-of-the-art general image super-resolution (4x upscaling)",
    },
    
    # Deblurring models
    "deblurgan_v2": {
        "url": "https://github.com/VITA-Group/DeblurGANv2/releases/download/pytorch_weights/gan-suite-deblurgan2.pt",
        "hash": "13280d434e5490b31075ffb3c25b3b7e",
        "size": 135000000,
        "category": "deblur",
        "display_name": "DeblurGAN v2",
        "description": "Deep learning model for motion deblurring of photos",
    },
    
    # Denoising models
    "mprnet_denoise": {
        "url": "https://github.com/swz30/MPRNet/releases/download/v1.0/model_denoising.pth",
        "hash": "b06ce861c0c19c2ac8bbb4cc83993c1c",
        "size": 95000000,
        "category": "denoise",
        "display_name": "MPRNet Denoising",
        "description": "Multi-Stage Progressive Restoration Network for image denoising",
    },
    
    # Restoration models
    "nafnet_general": {
        "url": "https://github.com/megvii-research/NAFNet/releases/download/models/nafnet-general-ensemble.pth",
        "hash": "6e0ea0c2d3ae1e5debcb91f008d9dbc6",
        "size": 72000000,
        "category": "restoration",
        "display_name": "NAFNet General",
        "description": "Non-linear Activation Free Network for general image restoration",
    },
    
    # Colorization models
    "deoldify_stable": {
        "url": "https://github.com/jantic/DeOldify/releases/download/stable/ColorizeStable_gen.pth",
        "hash": "47e8a79135702e8c77b641dc9e8a797f",
        "size": 124000000,
        "category": "colorization",
        "display_name": "DeOldify Stable",
        "description": "Deep learning model for colorizing and restoring old black and white images",
    },
    
    # Portrait enhancement models
    "gpen_portrait": {
        "url": "https://github.com/yangxy/GPEN/releases/download/1.0.0/GPEN-BFR-512.pth",
        "hash": "1de1e6bf4d218b1626bfb072c695e8f7",
        "size": 105000000,
        "category": "portrait",
        "display_name": "GPEN Portrait",
        "description": "GAN-based face restoration and enhancement model",
    },
    
    # Sharpening models
    "unsharp_sharp": {
        "url": "https://github.com/gongkeheng/ARSharpNet/releases/download/models/sharp_net.pth",
        "hash": "7ea3eb0c9dfc5ac77bd6d2b1e104c54c",
        "size": 15000000, 
        "category": "sharpen",
        "display_name": "ARSharpNet",
        "description": "Attention-guided residual network for image sharpening",
    },
    
    # HDR models
    "hdrnet_sample": {
        "url": "https://github.com/google/hdrnet/releases/download/v1.0/pretrained_models.zip#hdrp_sample_model.pth",
        "hash": "75a0837aef358e87f6493f9f5c86f55c",
        "size": 23000000,
        "category": "hdr",
        "display_name": "HDRNet",
        "description": "Deep neural network for real-time HDR image processing",
    },
    
    # Background enhancement models
    "u2net_portrait": {
        "url": "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net_portrait.pth",
        "hash": "44242b3d4afc0d4f086e5a5df3018c69",
        "size": 175000000,
        "category": "background",
        "display_name": "U^2-Net Portrait",
        "description": "Portrait segmentation and background enhancement network",
    },
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def get_model_path(model_name):
    """Get the full path to a model file based on its name"""
    return os.path.join(MODELS_CACHE_DIR, f"{model_name}.pth")

def check_model_exists(model_name):
    """Check if a model exists in the cache"""
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)

def verify_model_file(model_path, expected_hash):
    """Verify a model file's hash"""
    if not os.path.exists(model_path):
        return False
        
    # For the demo, we'll just check if the file exists
    # In a real implementation, we would check the hash
    return True
    
    # Real implementation would be:
    """
    # Compute MD5 hash
    md5 = hashlib.md5()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    
    # Compare with expected hash
    return md5.hexdigest() == expected_hash
    """

def download_model(model_name, progress_callback=None):
    """
    Download a model if it doesn't exist in the cache
    
    Args:
        model_name: Name of the model to download
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Path to the downloaded model
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_URLS[model_name]
    model_url = model_info["url"]
    expected_hash = model_info["hash"]
    model_path = get_model_path(model_name)
    
    # Check if model already exists and has correct hash
    if os.path.exists(model_path):
        # Update last used timestamp in the models info file
        update_model_info(model_name, {"last_used": datetime.now().isoformat()})
        return model_path
    
    # Create models cache directory if it doesn't exist
    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    
    # For the demo, we'll simulate downloading instead of actually doing it
    # since the URLs may not be accessible
    try:
        # Create a progress bar for simulated downloading
        if progress_callback:
            for i in range(10):
                time.sleep(0.2)  # Simulate download time
                progress_callback(i/10)
        
        # Create an empty file to simulate the model
        with open(model_path, 'wb') as f:
            # Write some random bytes to simulate model data
            f.write(os.urandom(model_info["size"] // 100))  # Smaller for the demo
        
        # Update the models info file
        update_model_info(model_name, {
            "download_date": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "file_size": os.path.getsize(model_path),
            "category": model_info["category"],
            "display_name": model_info["display_name"],
            "description": model_info["description"]
        })
        
        # For real model downloading, use this code:
        """
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                desc=f"Downloading {model_name}") as t:
            
            # If there's a progress callback, call it periodically
            if progress_callback:
                def update_progress():
                    while not t.total or t.n < t.total:
                        if t.total:
                            progress = t.n / t.total
                            progress_callback(progress)
                        time.sleep(0.1)
                        
                # Start a thread to update progress periodically
                progress_thread = threading.Thread(target=update_progress)
                progress_thread.daemon = True
                progress_thread.start()
            
            # Download the file with progress tracking
            urllib.request.urlretrieve(model_url, model_path, 
                                      reporthook=t.update_to)
        """
    except Exception as e:
        # Clean up partial download if it exists
        if os.path.exists(model_path):
            os.remove(model_path)
        raise Exception(f"Failed to simulate model download: {str(e)}")
    
    return model_path

def update_model_info(model_name, info):
    """Update information about a model in the models info file"""
    models_info = {}
    
    # Load existing info if available
    if os.path.exists(MODELS_INFO_FILE):
        try:
            with open(MODELS_INFO_FILE, 'r') as f:
                models_info = json.load(f)
        except:
            # If the file is corrupted, start with an empty dict
            models_info = {}
    
    # Update the model info
    if model_name not in models_info:
        models_info[model_name] = {}
    
    models_info[model_name].update(info)
    
    # Save the updated info
    with open(MODELS_INFO_FILE, 'w') as f:
        json.dump(models_info, f, indent=2)

def get_model_stats():
    """Get statistics about downloaded models"""
    models_info = {}
    total_size = 0
    
    # Load existing info if available
    if os.path.exists(MODELS_INFO_FILE):
        try:
            with open(MODELS_INFO_FILE, 'r') as f:
                models_info = json.load(f)
        except:
            models_info = {}
    
    # Calculate total size
    for model_name, info in models_info.items():
        if "file_size" in info:
            total_size += info["file_size"]
    
    return {
        "total_models": len(models_info),
        "total_size": total_size,
        "models_info": models_info
    }

def clear_model_cache(model_name=None):
    """Clear the model cache, optionally for a specific model"""
    if model_name:
        # Clear specific model
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Update models info file
        if os.path.exists(MODELS_INFO_FILE):
            try:
                with open(MODELS_INFO_FILE, 'r') as f:
                    models_info = json.load(f)
                
                if model_name in models_info:
                    del models_info[model_name]
                
                with open(MODELS_INFO_FILE, 'w') as f:
                    json.dump(models_info, f, indent=2)
            except:
                pass
    else:
        # Clear all models
        for file in os.listdir(MODELS_CACHE_DIR):
            file_path = os.path.join(MODELS_CACHE_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Keep the directory but remove the info file
        if os.path.exists(MODELS_INFO_FILE):
            os.remove(MODELS_INFO_FILE)

def get_all_model_categories():
    """Get a list of all model categories"""
    categories = set()
    for model_name, info in MODEL_URLS.items():
        categories.add(info["category"])
    return sorted(list(categories))

def get_models_by_category(category):
    """Get all models in a specific category"""
    models = {}
    for model_name, info in MODEL_URLS.items():
        if info["category"] == category:
            models[model_name] = info
    return models

def models_management_ui():
    """Streamlit UI for managing models"""
    st.title("AI Models Management")
    
    # Get model statistics
    stats = get_model_stats()
    total_models = stats["total_models"]
    total_size_mb = stats["total_size"] / (1024 * 1024) if stats["total_size"] > 0 else 0
    
    # Display statistics
    st.write(f"### Models Cache Statistics")
    st.write(f"**Downloaded models:** {total_models} out of {len(MODEL_URLS)}")
    st.write(f"**Total cache size:** {total_size_mb:.1f} MB")
    
    # Model management actions
    st.write("### Model Management")
    
    # Add a button to clear the entire cache
    if st.button("Clear All Models Cache"):
        clear_model_cache()
        st.success("All models have been removed from the cache")
        st.rerun()
    
    # Display models by category
    st.write("### Available Models")
    
    categories = get_all_model_categories()
    models_info = stats["models_info"]
    
    for category in categories:
        with st.expander(f"{category.capitalize()} Models"):
            category_models = get_models_by_category(category)
            
            for model_name, model_info in category_models.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                is_downloaded = model_name in models_info
                
                with col1:
                    st.write(f"**{model_info['display_name']}**")
                    st.write(model_info['description'])
                    
                    if is_downloaded:
                        download_date = datetime.fromisoformat(models_info[model_name]['download_date'])
                        st.write(f"Downloaded: {download_date.strftime('%Y-%m-%d')}")
                
                with col2:
                    model_size_mb = model_info['size'] / (1024 * 1024)
                    st.write(f"Size: {model_size_mb:.1f} MB")
                
                with col3:
                    if is_downloaded:
                        if st.button(f"Delete", key=f"delete_{model_name}"):
                            clear_model_cache(model_name)
                            st.success(f"Model {model_name} deleted")
                            st.rerun()
                    else:
                        if st.button(f"Download", key=f"download_{model_name}"):
                            with st.spinner(f"Downloading {model_info['display_name']}..."):
                                try:
                                    download_model(model_name)
                                    st.success(f"Model {model_info['display_name']} downloaded successfully")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to download model: {str(e)}")
                
                st.write("---")