# Image Enhancement App

A Streamlit application for enhancing images using state-of-the-art open-source AI models with interactive before/after comparison.

## Features

- **Super-resolution** - Enhance details and increase resolution with Real-ESRGAN
- **Deblurring** - Fix blurry or out-of-focus images with DeblurGANv2
- **Denoising** - Remove noise from low-light or grainy images with MPRNet
- **Image Restoration** - General restoration with NAFNet
- **Colorization** - Add color to black and white photos with DeOldify

## Usage

1. Upload an image
2. Select an enhancement model
3. Adjust model parameters
4. Click "Enhance Image"
5. View the before/after comparison
6. Download the enhanced image

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Technologies Used

- Streamlit
- OpenCV
- PyTorch
- NumPy
- Pillow