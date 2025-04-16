import streamlit as st
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

def get_image_download_link(img, filename="enhanced_image.png", text="Download Enhanced Image"):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    payload = buffered.getvalue()
    if len(payload) > 10 * 1024 * 1024:
        st.error("Image too large to encode for download.")
        return ""
    img_str = base64.b64encode(payload).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="download-btn">{text}</a>'
    return href

def create_comparison_image(original, enhanced):
    """
    Creates a side-by-side comparison of original and enhanced images.
    This is a placeholder - in the real implementation we would use a JavaScript-based
    comparison slider widget.
    
    Parameters:
    -----------
    original : PIL.Image
        The original image
    enhanced : PIL.Image
        The enhanced image
        
    Returns:
    --------
    PIL.Image
        A side-by-side comparison image
    """
    # Ensure both images are the same size
    original = original.convert("RGB")
    enhanced = enhanced.convert("RGB")
    
    # Resize enhanced image to match original if needed
    if original.size != enhanced.size:
        enhanced = enhanced.resize(original.size, Image.LANCZOS)
    
    # Get dimensions
    width, height = original.size
    
    # Create a new image with double the width for side-by-side comparison
    comparison = Image.new("RGB", (width * 2, height), (255, 255, 255))
    
    # Paste original and enhanced images
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced, (width, 0))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    
    # Add "BEFORE" and "AFTER" labels
    draw.rectangle([(10, 10), (110, 40)], fill=(0, 0, 0, 180))
    draw.text((20, 15), "BEFORE", fill=(255, 255, 255))
    
    draw.rectangle([(width + 10, 10), (width + 110, 40)], fill=(0, 0, 0, 180))
    draw.text((width + 20, 15), "AFTER", fill=(255, 255, 255))
    
    # Add a dividing line
    for y in range(0, height, 10):
        draw.rectangle([(width - 2, y), (width + 2, y + 5)], fill=(255, 255, 255))
    
    return comparison

def resize_image_if_needed(image, max_size=1024):
    """
    Resizes an image if it exceeds the maximum dimension.
    
    Parameters:
    -----------
    image : PIL.Image
        The image to resize
    max_size : int, optional
        The maximum allowed dimension
        
    Returns:
    --------
    PIL.Image
        The resized image if necessary, otherwise the original
    """
    width, height = image.size
    
    # Check if resizing is needed
    if width > max_size or height > max_size:
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Determine new dimensions
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)
        
        # Resize and return
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    # No resizing needed
    return image