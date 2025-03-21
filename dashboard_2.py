import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import requests
from io import BytesIO
from skimage.filters import rank
from skimage.morphology import disk
from skimage import img_as_ubyte

def get_image_from_url(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        st.warning(f"Could not retrieve image from {url}")
        return None

def rgb_to_grayscale(image):
    return image.convert('L')

def grayscale_to_binary(image, threshold=128):
    binary = image.point(lambda p: 255 if p > threshold else 0, '1')
    return binary

def rgb_to_binary(image, threshold=128):
    img_gray = image.convert('L')
    binary = img_gray.point(lambda p: 255 if p > threshold else 0, '1')
    return binary

def rgb_to_cmy(image):
    img_np = np.array(image) / 255.0
    cmy = 1 - img_np
    return Image.fromarray((cmy * 255).astype(np.uint8))

def rgb_to_hsi(image):
    img_np = np.array(image.convert('RGB')).astype(float) / 255.0
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    intensity = (r + g + b) / 3
    saturation = 1 - (3 / (r + g + b + 1e-6)) * np.minimum(r, np.minimum(g, b))
    hue = np.arccos(((r - g) + (r - b)) / (2 * np.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-6))
    hue = np.degrees(hue) % 360
    hsi = np.dstack((hue, saturation, intensity))
    return Image.fromarray((hsi * 255).astype(np.uint8))

def negative_transform(image):
    img_np = np.array(image)
    neg_img = 255 - img_np
    return Image.fromarray(neg_img)

def log_transform(image, c=1):
    img_np = np.array(image).astype(float) / 255.0 
    log_img = c * np.log(1 + img_np)
    log_img = (log_img / np.max(log_img)) * 255  
    return Image.fromarray(log_img.astype(np.uint8))


def gamma_transform(img, gamma):
    img_array = np.array(img) / 255.0  
    img_gamma = np.power(img_array, gamma) 
    img_gamma = (img_gamma * 255).astype(np.uint8)  
    return Image.fromarray(img_gamma)

# Histogram Processing
def calculate_histogram(image):
    hist = np.zeros(256)
    for pixel in image.getdata():
        hist[pixel] += 1
    return hist

def cumulative_histogram(hist):
    cum_hist = np.zeros(256)
    cum_hist[0] = hist[0]
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i-1] + hist[i]
    return cum_hist

def histogram_equalization(image):
    hist = calculate_histogram(image)
    cum_hist = cumulative_histogram(hist)
    
    total_pixels = image.width * image.height
    normalized_cum_hist = cum_hist * 255 / total_pixels
    normalized_cum_hist = normalized_cum_hist.astype(np.uint8)
    
    equalized = Image.new('L', image.size)
    pixels = image.getdata()
    new_pixels = [normalized_cum_hist[p] for p in pixels]
    equalized.putdata(new_pixels)
    return equalized

def histogram_specification(image, reference_image):
    hist_source = calculate_histogram(image)
    hist_ref = calculate_histogram(reference_image)

    cum_source = cumulative_histogram(hist_source)
    cum_ref = cumulative_histogram(hist_ref)

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        closest_match = np.argmin(np.abs(cum_source[i] - cum_ref))
        mapping[i] = closest_match

    new_pixels = [mapping[p] for p in image.getdata()]
    output_image = Image.new('L', image.size)
    output_image.putdata(new_pixels)
    return output_image

# Piecewise Linear Transformation Functions
def contrast_stretching(image, min_val=50, max_val=200):
    img_np = np.array(image, dtype=np.float32)
    img_np = np.clip(img_np, min_val, max_val)  
    stretched = 255 * (img_np - min_val) / (max_val - min_val)  
    return Image.fromarray(stretched.astype(np.uint8))

def intensity_level_slicing(image, min_val=100, max_val=200, preserve_intensity=True):
    img_np = np.array(image.convert('L'))  # Pastikan grayscale
    if preserve_intensity:
        sliced_img = np.where((img_np >= min_val) & (img_np <= max_val), img_np, 0)
    else:
        sliced_img = np.where((img_np >= min_val) & (img_np <= max_val), 255, 0)
    
    return Image.fromarray(sliced_img.astype(np.uint8))

def bit_plane_slicing(image, bit):
    img_np = np.array(image)
    bit_plane = (img_np & (1 << bit)) >> bit
    return Image.fromarray((bit_plane * 255).astype(np.uint8))

def local_histogram_processing(image, kernel_size=3):
    img_np = np.array(image)
    local_hist = rank.equalize(img_np, selem=disk(kernel_size))
    return Image.fromarray(img_as_ubyte(local_hist))

def histogram_dashboard():
    st.title('Image Processing Dashboard')
    
    # Upload Image
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    img_url = None if uploaded_img else st.text_input("Or enter image URL")
    
    if uploaded_img:
        img = Image.open(uploaded_img)
    elif img_url:
        img = get_image_from_url(img_url)
    else:
        img = None
    
    if img:
        img_gray = img.convert('L')

        st.subheader("Original Image")
        st.image(img, caption="Original Image", width=500)
        
        st.subheader("Select Operations")
        operations = st.multiselect(
            "Choose one or more operations",
            [
                "RGB to Binary", "RGB to CMY", "RGB to HSI",
                "Negative Transformation", "Log Transformation", "Gamma Transformation",
                "Histogram Equalization", "Bit-plane Slicing",
                "Contrast Stretching", "Intensity Level Slicing", "Local Histogram Processing"
            ]
        )

        # Menyimpan hasil transformasi
        processed_images = {}

        if "RGB to Binary" in operations:
            threshold = st.slider("Threshold", 0, 255, 128)
            binary_img = rgb_to_binary(img, threshold)
            processed_images["Binary Image"] = binary_img

        if "RGB to CMY" in operations:
            cmy_img = rgb_to_cmy(img)
            processed_images["CMY Image"] = cmy_img

        if "RGB to HSI" in operations:
            hsi_img = rgb_to_hsi(img)
            processed_images["HSI Image"] = hsi_img

        if "Negative Transformation" in operations:
            neg_img = negative_transform(img)
            processed_images["Negative Image"] = neg_img

        if "Log Transformation" in operations:
            c = st.slider("Constant (c)", 1, 10, 1)
            log_img = log_transform(img, c)
            processed_images["Log Transformed Image"] = log_img

        if "Gamma Transformation" in operations:
            gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0)
            gamma_img = gamma_transform(img, gamma)
            processed_images["Gamma Corrected Image"] = gamma_img

        if "Histogram Equalization" in operations:
            equalized = histogram_equalization(img_gray)
            processed_images["Equalized Image"] = equalized

        if "Bit-plane Slicing" in operations:
            bit = st.slider("Select Bit Plane (0-7)", 0, 7, 0)
            bit_img = bit_plane_slicing(img_gray, bit)
            processed_images[f"Bit-plane {bit} Image"] = bit_img

        if "Contrast Stretching" in operations:
            min_val = st.slider("Minimum Intensity", 0, 255, 50)
            max_val = st.slider("Maximum Intensity", 0, 255, 200)
            img_gray = img.convert("L") if img.mode != "L" else img_gray
            contrast_img = contrast_stretching(img_gray, min_val, max_val)
            processed_images["Contrast Stretched Image"] = contrast_img

        if "Intensity Level Slicing" in operations:
            min_val = st.slider("Minimum Intensity", 0, 255, 100)
            max_val = st.slider("Maximum Intensity", 0, 255, 200)
            preserve_intensity = st.checkbox("Preserve Original Intensity", value=True)
            sliced_img = intensity_level_slicing(img_gray, min_val, max_val, preserve_intensity)
            processed_images["Intensity Level Sliced Image"] = sliced_img


        if "Local Histogram Processing" in operations:
            kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
            local_hist_img = local_histogram_processing(img_gray, kernel_size)
            processed_images["Local Histogram Processed Image"] = local_hist_img


        # Menampilkan hasil transformasi
        for name, image in processed_images.items():
            st.subheader(name)
            st.image(image, caption=name, width=500)

if __name__ == "__main__":
    histogram_dashboard()
    
    
#     def histogram_dashboard():
#         st.title('Image Processing Dashboard')
    
#     # Upload Image
#     uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
#     img_url = None if uploaded_img else st.text_input("Or enter image URL")
    
#     if uploaded_img:
#         img = Image.open(uploaded_img)
#     elif img_url:
#         img = get_image_from_url(img_url)
#     else:
#         img = None
    
#     if img:
#         img_gray = img.convert('L')

#         st.subheader("Original Image")
#         st.image(img, caption="Original Image", width=500)
        
#         st.subheader("Select Operations")
#         operations = st.multiselect(
#             "Choose one or more operations",
#             [
#                 "RGB to Grayscale", "Grayscale to Binary", "RGB to Binary",
#                 "RGB to CMY", "RGB to HSI", "Negative Transformation", 
#                 "Log Transformation", "Gamma Transformation", "Histogram Equalization", 
#                 "Bit-plane Slicing", "Contrast Stretching", "Intensity Level Slicing", 
#                 "Local Histogram Processing", "Histogram Specification"
#             ]
#         )

#         # Menyimpan hasil transformasi
#         processed_images = {}

#         if "RGB to Grayscale" in operations:
#             gray_img = rgb_to_grayscale(img)
#             processed_images["Grayscale Image"] = gray_img

#         if "Grayscale to Binary" in operations:
#             threshold = st.slider("Threshold for Binary", 0, 255, 128)
#             binary_gray_img = grayscale_to_binary(img_gray, threshold)
#             processed_images["Binary Grayscale Image"] = binary_gray_img

#         if "RGB to Binary" in operations:
#             threshold = st.slider("Threshold", 0, 255, 128)
#             binary_img = rgb_to_binary(img, threshold)
#             processed_images["Binary Image"] = binary_img

#         if "RGB to CMY" in operations:
#             cmy_img = rgb_to_cmy(img)
#             processed_images["CMY Image"] = cmy_img

#         if "RGB to HSI" in operations:
#             hsi_img = rgb_to_hsi(img)
#             processed_images["HSI Image"] = hsi_img

#         if "Negative Transformation" in operations:
#             neg_img = negative_transform(img)
#             processed_images["Negative Image"] = neg_img

#         if "Log Transformation" in operations:
#             c = st.slider("Constant (c)", 1, 10, 1)
#             log_img = log_transform(img, c)
#             processed_images["Log Transformed Image"] = log_img

#         if "Gamma Transformation" in operations:
#             gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0)
#             gamma_img = gamma_transform(img, gamma)
#             processed_images["Gamma Corrected Image"] = gamma_img

#         if "Histogram Equalization" in operations:
#             equalized = histogram_equalization(img_gray)
#             processed_images["Equalized Image"] = equalized

#         if "Bit-plane Slicing" in operations:
#             bit = st.slider("Select Bit Plane (0-7)", 0, 7, 0)
#             bit_img = bit_plane_slicing(img_gray, bit)
#             processed_images[f"Bit-plane {bit} Image"] = bit_img

#         if "Contrast Stretching" in operations:
#             contrast_img = contrast_stretching(img_gray)
#             processed_images["Contrast Stretched Image"] = contrast_img

#         if "Intensity Level Slicing" in operations:
#             min_val = st.slider("Min Value", 0, 255, 100)
#             max_val = st.slider("Max Value", 0, 255, 200)
#             sliced_img = intensity_level_slicing(img_gray, min_val, max_val)
#             processed_images["Intensity Level Sliced Image"] = sliced_img

#         if "Local Histogram Processing" in operations:
#             kernel_size = st.slider("Kernel Size", 3, 11, 3, step=2)
#             local_hist_img = local_histogram_processing(img_gray, kernel_size)
#             processed_images["Local Histogram Processed Image"] = local_hist_img

#         if "Histogram Specification" in operations:
#             ref_img_upload = st.file_uploader("Upload Reference Image", type=['jpg', 'png', 'jpeg'])
#             if ref_img_upload:
#                 ref_img = Image.open(ref_img_upload).convert('L')
#                 spec_img = histogram_specification(img_gray, ref_img)
#                 processed_images["Histogram Specified Image"] = spec_img

#         # Menampilkan hasil transformasi
#         for name, image in processed_images.items():
#             st.subheader(name)
#             st.image(image, caption=name, width=500)

# if __name__ == "__main__":
#     histogram_dashboard()
