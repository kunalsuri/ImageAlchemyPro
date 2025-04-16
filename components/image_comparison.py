import streamlit as st
import base64
from PIL import Image
import io
import time
import random

def get_image_base64(img):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def image_comparison_component(img_before, img_after, label_before="BEFORE", label_after="AFTER"):
    """
    Create an interactive image comparison slider similar to CutOut.Pro
    
    Parameters:
    -----------
    img_before : PIL.Image
        The original image
    img_after : PIL.Image
        The enhanced image
    label_before : str
        Label for the before image
    label_after : str
        Label for the after image
    """
    # Ensure images are PIL Images
    if not isinstance(img_before, Image.Image):
        img_before = Image.fromarray(img_before)
    if not isinstance(img_after, Image.Image):
        img_after = Image.fromarray(img_after)
        
    # Ensure images are in RGB mode
    if img_before.mode != "RGB":
        img_before = img_before.convert("RGB")
    if img_after.mode != "RGB":
        img_after = img_after.convert("RGB")
    
    # Make sure both images are the same size
    width, height = img_before.size
    img_after = img_after.resize((width, height), Image.LANCZOS)
    
    # Convert images to base64 strings
    img_before_b64 = get_image_base64(img_before)
    img_after_b64 = get_image_base64(img_after)
    
    # Generate a unique ID for the component
    # Use timestamp plus random to avoid any ID collisions
    component_id = f"img-comp-{int(time.time())}-{random.randint(1000, 9999)}"
    
    # Simple HTML/CSS based slider with improved JavaScript
    html = f"""
    <style>
    .comparison-slider-wrapper {{
      position: relative;
      width: 100%;
      margin: 20px 0;
      overflow: hidden;
    }}
    .comparison-slider-wrapper img {{
      width: 100%;
      display: block;
    }}
    .comparison-slider {{
      position: relative;
      width: 100%;
      overflow: hidden;
      border-radius: 4px; 
      box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }}
    .comparison-item.overlay {{ 
      position: absolute; 
      top: 0; 
      left: 0; 
      height: 100%; 
      width: 50%; 
      overflow: hidden; 
      border-right: 2px solid white; 
    }}
    .comparison-item.underlay {{ 
      position: relative; 
    }}
    .slider-handle {{ 
      position: absolute; 
      top: 0; 
      bottom: 0;
      left: 50%; 
      width: 40px; 
      transform: translateX(-50%); 
      display: flex; 
      flex-direction: column; 
      align-items: center; 
      justify-content: center; 
      color: white; 
      font-weight: bold; 
      cursor: ew-resize; 
      z-index: 10;
    }}
    .slider-handle::before {{ 
      content: ""; 
      position: absolute; 
      top: 0; 
      bottom: 0; 
      left: 50%; 
      width: 4px; 
      background: white; 
      transform: translateX(-50%); 
      z-index: -1;
    }}
    .slider-handle::after {{ 
      content: ""; 
      position: absolute; 
      top: calc(50% - 20px); 
      height: 40px; 
      width: 40px; 
      background: #4F8BF9; 
      border: 3px solid white; 
      border-radius: 50%; 
      z-index: -1;
    }}
    .slider-handle svg {{ 
      width: 24px; 
      height: 24px; 
      position: absolute; 
      left: 8px;
      pointer-events: none;
    }}
    .slider-label {{ 
      position: absolute; 
      top: 15px; 
      padding: 5px 10px; 
      background-color: rgba(0,0,0,0.5); 
      color: white; 
      font-weight: bold; 
      font-size: 14px; 
      border-radius: 4px; 
      z-index: 10;
    }}
    .label-before {{ 
      left: 15px; 
    }}
    .label-after {{ 
      right: 15px; 
    }}
    </style>

    <div class="comparison-slider-wrapper">
        <div id="slider-{component_id}" class="comparison-slider">
            <div class="comparison-item underlay">
                <div class="slider-label label-after">{label_after}</div>
                <img src="{img_after_b64}" alt="{label_after}">
            </div>
            <div id="overlay-{component_id}" class="comparison-item overlay">
                <div class="slider-label label-before">{label_before}</div>
                <img src="{img_before_b64}" alt="{label_before}">
            </div>
            <div id="handle-{component_id}" class="slider-handle">
                <svg viewBox="0 0 24 24">
                    <path fill="white" d="M8.59,16.58L13.17,12L8.59,7.41L10,6L16,12L10,18L8.59,16.58Z" />
                    <path fill="white" d="M15.41,16.58L20,12L15.41,7.41L14,6L20,12L14,18L15.41,16.58Z" 
                          transform="rotate(180 17 12)" />
                </svg>
            </div>
        </div>
    </div>

    <script>
    // Immediately executing function to initialize the slider
    (function() {{
      // Add a small delay to ensure elements are available in the DOM
      setTimeout(function() {{
        // Get the elements
        var slider = document.getElementById("slider-{component_id}");
        var overlay = document.getElementById("overlay-{component_id}");
        var handle = document.getElementById("handle-{component_id}");
        
        // Check if elements exist
        if (!slider || !overlay || !handle) return;
        
        // Initialize drag state
        var isDragging = false;
        
        // Event handlers
        function startDrag(e) {{
          isDragging = true;
          e.preventDefault();
        }}
        
        function endDrag() {{
          isDragging = false;
        }}
        
        function drag(e) {{
          if (!isDragging) return;
          
          var clientX;
          if (e.type === 'touchmove') {{
            clientX = e.touches[0].clientX;
          }} else {{
            clientX = e.clientX;
          }}
          
          var sliderRect = slider.getBoundingClientRect();
          var sliderWidth = sliderRect.width;
          var offsetX = clientX - sliderRect.left;
          
          // Calculate percentage
          var percent = (offsetX / sliderWidth) * 100;
          percent = Math.min(Math.max(percent, 0), 100);
          
          // Update overlay and handle position
          overlay.style.width = percent + "%";
          handle.style.left = percent + "%";
        }}
        
        // Mouse events
        handle.addEventListener('mousedown', startDrag);
        document.addEventListener('mouseup', endDrag);
        document.addEventListener('mousemove', drag);
        
        // Touch events
        handle.addEventListener('touchstart', startDrag);
        document.addEventListener('touchend', endDrag);
        document.addEventListener('touchmove', drag);
        
        // Set initial position
        overlay.style.width = "50%";
        handle.style.left = "50%";
      }}, 100); // 100ms delay should be enough for elements to load
    }})();
    </script>
    """
    
    # Inject the HTML into Streamlit
    st.markdown(html, unsafe_allow_html=True)