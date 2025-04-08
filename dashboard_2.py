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

def calculate_histogram_rgb(image):
    img_np = np.array(image)
    
    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)
    
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            hist_r[img_np[i, j, 0]] += 1
            hist_g[img_np[i, j, 1]] += 1
            hist_b[img_np[i, j, 2]] += 1
            
    return hist_r, hist_g, hist_b

def cumulative_histogram_rgb(hist_r, hist_g, hist_b):
    cum_hist_r = np.zeros(256)
    cum_hist_g = np.zeros(256)
    cum_hist_b = np.zeros(256)
    
    cum_hist_r[0] = hist_r[0]
    cum_hist_g[0] = hist_g[0]
    cum_hist_b[0] = hist_b[0]
    
    for i in range(1, 256):
        cum_hist_r[i] = cum_hist_r[i-1] + hist_r[i]
        cum_hist_g[i] = cum_hist_g[i-1] + hist_g[i]
        cum_hist_b[i] = cum_hist_b[i-1] + hist_b[i]
        
    return cum_hist_r, cum_hist_g, cum_hist_b

def histogram_equalization_rgb(image):
    img_np = np.array(image)
    height, width, _ = img_np.shape
    total_pixels = height * width
    
    hist_r, hist_g, hist_b = calculate_histogram_rgb(image)
    
    cum_hist_r, cum_hist_g, cum_hist_b = cumulative_histogram_rgb(hist_r, hist_g, hist_b)
    
    norm_cum_hist_r = (cum_hist_r * 255 / total_pixels).astype(np.uint8)
    norm_cum_hist_g = (cum_hist_g * 255 / total_pixels).astype(np.uint8)
    norm_cum_hist_b = (cum_hist_b * 255 / total_pixels).astype(np.uint8)
    
    equalized_img = np.zeros_like(img_np)
    
    for i in range(height):
        for j in range(width):
            equalized_img[i, j, 0] = norm_cum_hist_r[img_np[i, j, 0]]
            equalized_img[i, j, 1] = norm_cum_hist_g[img_np[i, j, 1]]
            equalized_img[i, j, 2] = norm_cum_hist_b[img_np[i, j, 2]]
    
    return Image.fromarray(equalized_img)

def histogram_stretching_rgb(image, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    img_np = np.array(image)
    height, width, _ = img_np.shape
    
    stretched_img = np.zeros_like(img_np)
    
    if r_min == 0 and r_max == 255:
        r_min_actual = np.min(img_np[:,:,0])
        r_max_actual = np.max(img_np[:,:,0])
    else:
        r_min_actual, r_max_actual = r_min, r_max
        
    if g_min == 0 and g_max == 255:
        g_min_actual = np.min(img_np[:,:,1])
        g_max_actual = np.max(img_np[:,:,1])
    else:
        g_min_actual, g_max_actual = g_min, g_max
        
    if b_min == 0 and b_max == 255:
        b_min_actual = np.min(img_np[:,:,2])
        b_max_actual = np.max(img_np[:,:,2])
    else:
        b_min_actual, b_max_actual = b_min, b_max
    
    r_range = r_max_actual - r_min_actual
    g_range = g_max_actual - g_min_actual
    b_range = b_max_actual - b_min_actual
    
    r_range = 1 if r_range == 0 else r_range
    g_range = 1 if g_range == 0 else g_range
    b_range = 1 if b_range == 0 else b_range
    
    stretched_img[:,:,0] = np.clip((img_np[:,:,0] - r_min_actual) * (255.0 / r_range), 0, 255).astype(np.uint8)
    stretched_img[:,:,1] = np.clip((img_np[:,:,1] - g_min_actual) * (255.0 / g_range), 0, 255).astype(np.uint8)
    stretched_img[:,:,2] = np.clip((img_np[:,:,2] - b_min_actual) * (255.0 / b_range), 0, 255).astype(np.uint8)
    
    return Image.fromarray(stretched_img)

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
    local_hist = rank.equalize(img_np, footprint=disk(kernel_size))
    return Image.fromarray(img_as_ubyte(local_hist))

def plot_histogram(image, title="Histogram"):
    if image.mode == '1':  
        img_array = np.array(image.convert('L')).flatten()
    elif image.mode == 'RGB':
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        
        for i, color in enumerate(colors):
            img_array = np.array(image)[:,:,i].flatten()
            ax[i].hist(img_array, bins=256, color=color, alpha=0.7)
            ax[i].set_title(f'{color.capitalize()} Channel')
            ax[i].set_xlim([0, 255])
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    else:
        img_array = np.array(image).flatten()
    
    if image.mode != 'RGB':
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(img_array, bins=256, color='gray', alpha=0.7)
        ax.set_title(title)
        ax.set_xlim([0, 255])
        plt.tight_layout()
        return fig

def plot_histogram_comparison(original, processed, title_orig="Original", title_proc="Processed"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if original.mode == '1':  
        img_array_orig = np.array(original.convert('L')).flatten()
    else:
        if original.mode != 'L':
            original = original.convert('L')
        img_array_orig = np.array(original).flatten()
    
    if processed.mode == '1':  
        img_array_proc = np.array(processed.convert('L')).flatten()
    else:
        if processed.mode != 'L':
            processed = processed.convert('L')
        img_array_proc = np.array(processed).flatten()
    
    ax1.hist(img_array_orig, bins=256, color='blue', alpha=0.7)
    ax1.set_title(title_orig)
    ax1.set_xlim([0, 255])
    
    ax2.hist(img_array_proc, bins=256, color='red', alpha=0.7)
    ax2.set_title(title_proc)
    ax2.set_xlim([0, 255])
    
    plt.tight_layout()
    return fig

def plot_rgb_histograms_comparison(original, processed, title_orig="Original", title_proc="Processed"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    orig_np = np.array(original)
    axes[0, 0].hist(orig_np[:,:,0].flatten(), bins=256, color='red', alpha=0.7)
    axes[0, 0].set_title(f"{title_orig} - Red Channel")
    axes[0, 0].set_xlim([0, 255])
    
    axes[0, 1].hist(orig_np[:,:,1].flatten(), bins=256, color='green', alpha=0.7)
    axes[0, 1].set_title(f"{title_orig} - Green Channel")
    axes[0, 1].set_xlim([0, 255])
    
    axes[0, 2].hist(orig_np[:,:,2].flatten(), bins=256, color='blue', alpha=0.7)
    axes[0, 2].set_title(f"{title_orig} - Blue Channel")
    axes[0, 2].set_xlim([0, 255])
    
    proc_np = np.array(processed)
    axes[1, 0].hist(proc_np[:,:,0].flatten(), bins=256, color='red', alpha=0.7)
    axes[1, 0].set_title(f"{title_proc} - Red Channel")
    axes[1, 0].set_xlim([0, 255])
    
    axes[1, 1].hist(proc_np[:,:,1].flatten(), bins=256, color='green', alpha=0.7)
    axes[1, 1].set_title(f"{title_proc} - Green Channel")
    axes[1, 1].set_xlim([0, 255])
    
    axes[1, 2].hist(proc_np[:,:,2].flatten(), bins=256, color='blue', alpha=0.7)
    axes[1, 2].set_title(f"{title_proc} - Blue Channel")
    axes[1, 2].set_xlim([0, 255])
    
    plt.tight_layout()
    return fig

def histogram_dashboard():
    st.title('Image Processing Dashboard')
    st.header("Upload Image")
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    img_url = None if uploaded_img else st.text_input("Or enter image URL")
        
    with st.sidebar:
       
        st.header("Operations")
        operations = st.multiselect(
            "Choose one or more operations",
            [
                "RGB to Binary", "RGB to CMY", "RGB to HSI",
                "Negative Transformation", "Log Transformation", "Gamma Transformation",
                "Bit-plane Slicing","Contrast Stretching", "Intensity Level Slicing", 
                "Local Histogram Processing", "Histogram Equalization", "Histogram Stretching"
            ]
        )

        operation_params = {}
        
        if "RGB to Binary" in operations:
            operation_params["threshold"] = st.slider("Binary Threshold", 0, 255, 128)
            
        if "Log Transformation" in operations:
            operation_params["c"] = st.slider("Log Constant (c)", 1, 10, 1)
            
        if "Gamma Transformation" in operations:
            operation_params["gamma"] = st.slider("Gamma Value", 0.1, 3.0, 1.0)
            
        if "Histogram Stretching" in operations:
            st.subheader("RGB Histogram Stretching")
            operation_params["auto_detect"] = st.checkbox("Auto-detect min/max values", value=True)
            
            if not operation_params["auto_detect"]:
                col1, col2 = st.columns(2)
                with col1:
                    operation_params["r_min"] = st.slider("Red Min", 0, 255, 0)
                    operation_params["g_min"] = st.slider("Green Min", 0, 255, 0)
                    operation_params["b_min"] = st.slider("Blue Min", 0, 255, 0)
                
                with col2:
                    operation_params["r_max"] = st.slider("Red Max", 0, 255, 255)
                    operation_params["g_max"] = st.slider("Green Max", 0, 255, 255)
                    operation_params["b_max"] = st.slider("Blue Max", 0, 255, 255)
            
        if "Bit-plane Slicing" in operations:
            operation_params["bit"] = st.slider("Select Bit Plane (0-7)", 0, 7, 0)
            
        if "Contrast Stretching" in operations:
            operation_params["min_val_contrast"] = st.slider("Min Intensity (Contrast)", 0, 255, 50)
            operation_params["max_val_contrast"] = st.slider("Max Intensity (Contrast)", 0, 255, 200)
            
        if "Intensity Level Slicing" in operations:
            operation_params["min_val_slice"] = st.slider("Min Intensity (Slice)", 0, 255, 100)
            operation_params["max_val_slice"] = st.slider("Max Intensity (Slice)", 0, 255, 200)
            operation_params["preserve_intensity"] = st.checkbox("Preserve Original Intensity", value=True)
            
        if "Local Histogram Processing" in operations:
            operation_params["kernel_size"] = st.slider("Kernel Size", 3, 15, 3, step=2)
    
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
        st.image(img, caption="Original Image", width=500)
        
        st.subheader("Original Image Histogram")
        st.pyplot(plot_histogram(img, "Original RGB Histogram"))
        
        processed_images = {}
        comparisons = []  

        if "RGB to Binary" in operations:
            binary_img = rgb_to_binary(img, operation_params["threshold"])
            processed_images["Binary Image"] = binary_img

        if "RGB to CMY" in operations:
            cmy_img = rgb_to_cmy(img)
            processed_images["CMY Image"] = cmy_img
            comparisons.append(("RGB Channel Comparison: Original vs CMY Image", plot_rgb_histograms_comparison(img, cmy_img, "Original", "CMY Image")))

        if "RGB to HSI" in operations:
            hsi_img = rgb_to_hsi(img)
            processed_images["HSI Image"] = hsi_img
            comparisons.append(("RGB Channel Comparison: Original vs HSI Image", plot_rgb_histograms_comparison(img, hsi_img, "Original", "HSI Image")))

        if "Negative Transformation" in operations:
            neg_img = negative_transform(img)
            processed_images["Negative Image"] = neg_img
            comparisons.append(("RGB Channel Comparison: Original vs Negative Image", plot_rgb_histograms_comparison(img, neg_img, "Original", "Negative Image")))

        if "Log Transformation" in operations:
            log_img = log_transform(img, operation_params["c"])
            processed_images["Log Transformed Image"] = log_img
            comparisons.append(("RGB Channel Comparison: Original vs Log Transformed Image", plot_rgb_histograms_comparison(img, log_img, "Original", "Log Transformed Image")))

        if "Gamma Transformation" in operations:
            gamma_img = gamma_transform(img, operation_params["gamma"])
            processed_images["Gamma Corrected Image"] = gamma_img
            comparisons.append(("RGB Channel Comparison: Original vs Gamma Corrected Image", plot_rgb_histograms_comparison(img, gamma_img, "Original", "Gamma Corrected Image")))

        if "Histogram Equalization" in operations:
            equalized = histogram_equalization_rgb(img)
            processed_images["Equalized RGB Image"] = equalized
            comparisons.append(("RGB Channel Comparison: Original vs Equalized RGB Image", plot_rgb_histograms_comparison(img, equalized, "Original", "Equalized RGB Image")))

        if "Histogram Stretching" in operations:
            if operation_params["auto_detect"]:
                stretched_img = histogram_stretching_rgb(img)
            else:
                stretched_img = histogram_stretching_rgb(
                    img, 
                    operation_params["r_min"], 
                    operation_params["r_max"], 
                    operation_params["g_min"], 
                    operation_params["g_max"], 
                    operation_params["b_min"], 
                    operation_params["b_max"]
                )
                
            processed_images["Stretched RGB Image"] = stretched_img
            comparisons.append(("RGB Channel Comparison: Original vs Stretched RGB Image", plot_rgb_histograms_comparison(img, stretched_img, "Original", "Stretched RGB Image")))

        if "Bit-plane Slicing" in operations:
            bit_img = bit_plane_slicing(img_gray, operation_params["bit"])
            processed_images[f"Bit-plane {operation_params['bit']} Image"] = bit_img
            comparisons.append((f"Histogram Comparison: Original vs Bit-plane {operation_params['bit']} Image", plot_histogram_comparison(img_gray, bit_img, "Original (Grayscale)", f"Bit-plane {operation_params['bit']} Image")))

        if "Contrast Stretching" in operations:
            img_gray = img.convert("L") if img.mode != "L" else img_gray
            contrast_img = contrast_stretching(img_gray, operation_params["min_val_contrast"], operation_params["max_val_contrast"])
            processed_images["Contrast Stretched Image"] = contrast_img
            comparisons.append(("Histogram Comparison: Original vs Contrast Stretched Image", plot_histogram_comparison(img_gray, contrast_img, "Original (Grayscale)", "Contrast Stretched Image")))

        if "Intensity Level Slicing" in operations:
            sliced_img = intensity_level_slicing(img_gray, operation_params["min_val_slice"], operation_params["max_val_slice"], operation_params["preserve_intensity"])
            processed_images["Intensity Level Sliced Image"] = sliced_img
            comparisons.append(("Histogram Comparison: Original vs Intensity Level Sliced Image", plot_histogram_comparison(img_gray, sliced_img, "Original (Grayscale)", "Intensity Level Sliced Image")))

        if "Local Histogram Processing" in operations:
            local_hist_img = local_histogram_processing(img_gray, operation_params["kernel_size"])
            processed_images["Local Histogram Processed Image"] = local_hist_img
            comparisons.append(("Histogram Comparison: Original vs Local Histogram Processed Image", plot_histogram_comparison(img_gray, local_hist_img, "Original (Grayscale)", "Local Histogram Processed Image")))

        for name, image in processed_images.items():
            st.subheader(name)
            st.image(image, caption=name, width=500)
            
            st.subheader(f"{name} Histogram")
            st.pyplot(plot_histogram(image, f"{name} Histogram"))
        
        if comparisons:
            st.header("Histogram Comparisons")
            for title, plot in comparisons:
                st.subheader(title)
                st.pyplot(plot)

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
