import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import requests
from io import BytesIO
from skimage.filters import gaussian
from skimage import img_as_ubyte
from scipy import ndimage

def get_image_from_url(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        st.warning(f"Could not retrieve image from {url}")
        return None

def apply_sobel_filter(image):
    img_gray = np.array(image.convert('L')).astype(float)
    
    sobel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])
    
    grad_x = ndimage.convolve(img_gray, sobel_x)
    grad_y = ndimage.convolve(img_gray, sobel_y)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    return Image.fromarray(gradient)

def apply_prewitt_filter(image):
    img_gray = np.array(image.convert('L')).astype(float)
    
    prewitt_x = np.array([[-1, 0, 1], 
                           [-1, 0, 1], 
                           [-1, 0, 1]])
    
    prewitt_y = np.array([[1, 1, 1], 
                           [0, 0, 0], 
                           [-1, -1, -1]])
    
    grad_x = ndimage.convolve(img_gray, prewitt_x)
    grad_y = ndimage.convolve(img_gray, prewitt_y)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    return Image.fromarray(gradient)

def apply_roberts_filter(image):
    img_gray = np.array(image.convert('L')).astype(float)
    
    roberts_x = np.array([[1, 0], 
                          [0, -1]])
    
    roberts_y = np.array([[0, 1], 
                          [-1, 0]])
    
    grad_x = ndimage.convolve(img_gray, roberts_x)
    grad_y = ndimage.convolve(img_gray, roberts_y)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    return Image.fromarray(gradient)

def apply_compass_filter(image, direction='all'):
    img_gray = np.array(image.convert('L')).astype(float)
    
    compass_filters = {
        'n': np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        'nw': np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        'w': np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        'sw': np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        's': np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        'se': np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        'e': np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        'ne': np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    }
    
    if direction != 'all':
        gradient = ndimage.convolve(img_gray, compass_filters[direction])
        gradient = np.abs(gradient)
    else:
        responses = []
        for dir_name, kernel in compass_filters.items():
            response = ndimage.convolve(img_gray, kernel)
            responses.append(response)
        
        gradient = np.max(np.array(responses), axis=0)
    
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    return Image.fromarray(gradient)

def add_noise(image, noise_type='gaussian', intensity=0.05):
    img_array = np.array(image).astype(float) / 255.0
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, img_array.shape)
        noisy_img = img_array + noise
    elif noise_type == 'salt_pepper':
        noisy_img = img_array.copy()
        salt_mask = np.random.random(img_array.shape) < intensity/2
        noisy_img[salt_mask] = 1.0
        pepper_mask = np.random.random(img_array.shape) < intensity/2
        noisy_img[pepper_mask] = 0.0
    
    noisy_img = np.clip(noisy_img, 0, 1)
    
    return Image.fromarray((noisy_img * 255).astype(np.uint8))

def apply_log_filter(image, sigma=1.0):
    img_gray = np.array(image.convert('L')).astype(float)
    
    img_smooth = gaussian(img_gray, sigma=sigma)
    
    laplacian = np.array([[0, -1, 0], 
                          [-1, 4, -1], 
                          [0, -1, 0]])
    
    edges = ndimage.convolve(img_smooth, laplacian)
    edges = np.abs(edges)
    edges = (edges / edges.max() * 255).astype(np.uint8)
    
    return Image.fromarray(edges)

def edge_detection_dashboard():
    st.title('Edge Detection')
    
    st.header("Image Input")
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    img_url = None if uploaded_img else st.text_input("Or enter image URL")
    with st.sidebar:
       
        
        st.header("Edge Detection Filters")
        edge_operations = st.multiselect(
            "Filtering Methods",
            [
                "Sobel Filter", 
                "Prewitt Filter", 
                "Roberts Filter", 
                "Compass Filter", 
                "LoG Filter (Noisy Image)"
            ]
        )
        
        compass_direction = None
        if "Compass Filter" in edge_operations:
            compass_direction = st.selectbox(
                "Choose compass direction",
                ["all", "n", "nw", "w", "sw", "s", "se", "e", "ne"]
            )
        
        noise_params = {}
        if "LoG Filter (Noisy Image)" in edge_operations:
            st.subheader("Noise and LoG Parameters")
            noise_params['noise_type'] = st.selectbox("Noise Type", ["gaussian", "salt_pepper"])
            noise_params['noise_intensity'] = st.slider("Noise Intensity", 0.01, 0.3, 0.05)
            noise_params['sigma'] = st.slider("Gaussian Sigma", 0.5, 5.0, 1.0)
    
    if uploaded_img:
        img = Image.open(uploaded_img)
    elif img_url:
        img = get_image_from_url(img_url)
    else:
        img = None
    
    if img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_gray = img.convert('L')

        st.subheader("Original Image")
        st.image(img, caption="Original Image")
        
        processed_images = {}
        
        if "Sobel Filter" in edge_operations:
            sobel_img = apply_sobel_filter(img)
            processed_images["Sobel Edge Detection"] = sobel_img
        
        if "Prewitt Filter" in edge_operations:
            prewitt_img = apply_prewitt_filter(img)
            processed_images["Prewitt Edge Detection"] = prewitt_img
        
        if "Roberts Filter" in edge_operations:
            roberts_img = apply_roberts_filter(img)
            processed_images["Roberts Edge Detection"] = roberts_img
        
        if "Compass Filter" in edge_operations:
            compass_img = apply_compass_filter(img, compass_direction)
            processed_images[f"Compass Edge Detection ({compass_direction})"] = compass_img
        
        if "LoG Filter (Noisy Image)" in edge_operations:
            noisy_img = add_noise(img, noise_params['noise_type'], noise_params['noise_intensity'])
            processed_images["Noisy Image"] = noisy_img
            
            log_img = apply_log_filter(noisy_img, noise_params['sigma'])
            processed_images["LoG Edge Detection"] = log_img
        
        for name, image in processed_images.items():
            st.subheader(name)
            st.image(image, caption=name)

        if len(edge_operations) > 1:
            st.subheader("Comparison of Edge Detection Methods")
            
            methods = [name for name in processed_images.keys() if name != "Noisy Image"]
            num_methods = len(methods)
            
            if num_methods > 0:
                cols = st.columns(min(num_methods, 3))
                for i, method_name in enumerate(methods):
                    with cols[i % 3]:
                        st.image(processed_images[method_name], caption=method_name)

if __name__ == "__main__":
    edge_detection_dashboard()