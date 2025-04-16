import numpy as np
import cv2
import torch
import os
import time
from PIL import Image
import warnings
import streamlit as st

from model_manager import download_model, check_model_exists

# Global cache for loaded models
loaded_models = {}

def get_available_models():
    """
    Returns a dictionary of available image enhancement models.
    """
    return {
        'superres': {
            'display_name': 'Super Resolution',
            'description': 'Increase image resolution and clarity',
            'models': {
                'real_esrgan_x4': 'Real-ESRGAN x4'
            }
        },
        'deblur': {
            'display_name': 'Deblur',
            'description': 'Remove blur from photos',
            'models': {
                'deblurgan_v2': 'DeblurGAN v2'
            }
        },
        'denoise': {
            'display_name': 'Denoise',
            'description': 'Remove noise from images',
            'models': {
                'mprnet_denoise': 'MPRNet Denoise'
            }
        },
        'restoration': {
            'display_name': 'Restoration',
            'description': 'General image restoration and enhancement',
            'models': {
                'nafnet_general': 'NAFNet General'
            }
        },
        'colorization': {
            'display_name': 'Colorize',
            'description': 'Add color to black and white images',
            'models': {
                'deoldify_stable': 'DeOldify'
            }
        },
        'sharpen': {
            'display_name': 'Sharpen',
            'description': 'Enhance image sharpness and details',
            'models': {
                'unsharp_sharp': 'ARSharpNet'
            }
        },
        'portrait': {
            'display_name': 'Portrait Enhancement',
            'description': 'Enhance facial features and skin tones',
            'models': {
                'gpen_portrait': 'GPEN Portrait'
            }
        },
        'hdr': {
            'display_name': 'HDR Effect',
            'description': 'Enhance colors, contrast, and dynamic range',
            'models': {
                'hdrnet_sample': 'HDRNet'
            }
        },
        'background': {
            'display_name': 'Background Enhancement',
            'description': 'Enhance or blur image background',
            'models': {
                'u2net_portrait': 'U²-Net Portrait'
            }
        }
    }

def load_model(model_name, model_type):
    """
    Cache model loading to avoid reloading on each run.
    
    Args:
        model_name: Name of the model to load
        model_type: Type of the model (superres, deblur, etc.)
        
    Returns:
        Loaded model object
    """
    global loaded_models
    
    # If model is already loaded, return from cache
    model_key = f"{model_type}_{model_name}"
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    # Placeholder loading logic - real implementation would load actual models
    # For now, just return the model name since our enhancement functions will 
    # just simulate the effects
    
    # Check if model exists in cache, if not download it
    if not check_model_exists(model_name):
        download_model(model_name)
    
    # Simulated model loading
    loaded_models[model_key] = {
        'name': model_name,
        'type': model_type,
        'loaded_time': time.time()
    }
    
    return loaded_models[model_key]

def enhance_image_superres(image, model_config, params, progress_callback=None):
    """
    Super-resolution enhancement using Real-ESRGAN or similar models.
    """
    # In an actual implementation, this would use the real model
    # For this demo, we'll simulate the effect
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Simple upscaling simulation using OpenCV
    h, w = img_np.shape[:2]
    scale = params.get('scale', 2.0)
    upscaled = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Apply some sharpening to simulate super-resolution
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(upscaled, -1, kernel)
    
    # Convert back to PIL image
    return Image.fromarray(enhanced)

def enhance_image_deblur(image, model_config, params, progress_callback=None):
    """
    Deblurring using DeblurGANv2 or similar models.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple deblurring simulation with unsharp mask
    gaussian = cv2.GaussianBlur(img_np, (0, 0), 3)
    enhanced = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)
    
    return Image.fromarray(enhanced)

def enhance_image_denoise(image, model_config, params, progress_callback=None):
    """
    Denoising using MPRNet or similar models.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple denoising simulation with bilateral filter
    strength = params.get('strength', 15)
    enhanced = cv2.bilateralFilter(img_np, 9, strength, strength)
    
    return Image.fromarray(enhanced)

def enhance_image_restoration(image, model_config, params, progress_callback=None):
    """
    General restoration using NAFNet or similar models.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple restoration simulation with color enhancement and sharpening
    # Increase saturation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Normalize brightness
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.5)
    
    return Image.fromarray(enhanced)

def enhance_image_colorization(image, model_config, params, progress_callback=None):
    """
    Colorization using DeOldify or similar models.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array and ensure grayscale
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # If the image is already colored, just enhance colors
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.5  # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        # For actual grayscale images, we'd use the model
        # For simulation, just add a sepia tone
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        enhanced = np.zeros((*gray.shape, 3), dtype=np.uint8)
        
        # Create sepia effect
        enhanced[:, :, 0] = np.clip(gray * 0.189 + 100, 0, 255)  # Blue
        enhanced[:, :, 1] = np.clip(gray * 0.349 + 50, 0, 255)   # Green
        enhanced[:, :, 2] = np.clip(gray * 0.486 + 20, 0, 255)   # Red
    
    return Image.fromarray(enhanced)

def enhance_image_sharpen(image, model_config, params, progress_callback=None):
    """
    Sharpening using advanced techniques.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple sharpening with unsharp mask
    strength = params.get('strength', 1.5)
    blur = cv2.GaussianBlur(img_np, (0, 0), 3)
    enhanced = cv2.addWeighted(img_np, strength, blur, 1-strength, 0)
    
    return Image.fromarray(enhanced)

def enhance_image_portrait(image, model_config, params, progress_callback=None):
    """
    Portrait enhancement for improving facial features and skin tones.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple portrait enhancement with color and skin smoothing
    # Face detection would be used in a real implementation
    
    # Enhance skin tones
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    # Create a mask for skin tones
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply bilateral filter to smooth skin while preserving edges
    smoothed = cv2.bilateralFilter(img_np, 9, 75, 75)
    
    # Combine original image with smoothed image using the mask
    skin_mask_3d = np.stack([skin_mask] * 3, axis=2) / 255.0
    enhanced = img_np * (1 - skin_mask_3d) + smoothed * skin_mask_3d
    
    # Enhance overall image
    enhanced = cv2.addWeighted(enhanced.astype(np.uint8), 1.1, cv2.GaussianBlur(enhanced.astype(np.uint8), (0, 0), 3), -0.1, 0)
    
    return Image.fromarray(enhanced.astype(np.uint8))

def enhance_image_hdr(image, model_config, params, progress_callback=None):
    """
    Apply HDR effect to enhance colors, contrast, and details.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple HDR simulation
    # Split into RGB channels
    b, g, r = cv2.split(img_np)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    
    # Merge back
    enhanced = cv2.merge([b, g, r])
    
    # Increase details with sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.5)
    
    # Increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.3
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(enhanced)

def enhance_image_background(image, model_config, params, progress_callback=None):
    """
    Enhance or blur image background while preserving the subject.
    """
    # Simulate processing
    if progress_callback:
        for i in range(10):
            time.sleep(0.1)
            progress_callback(i/10)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Simple background blur simulation
    # In a real implementation, we would use segmentation model to detect subjects
    
    # Create a center-weighted mask (assuming the subject is in the center)
    h, w = img_np.shape[:2]
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    
    # Create a distance map from the center
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Normalize to 0-1 range (closer to center = 1, far = 0)
    radius = min(h, w) * 0.4  # Adjust for subject size
    mask = np.clip(1 - dist_from_center / radius, 0, 1)
    
    # Apply gaussian blur to background
    blurred = cv2.GaussianBlur(img_np, (51, 51), 0)
    
    # Combine original with blurred using the mask
    mask_3d = np.stack([mask] * 3, axis=2)
    enhanced = img_np * mask_3d + blurred * (1 - mask_3d)
    
    return Image.fromarray(enhanced.astype(np.uint8))