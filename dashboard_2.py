import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import requests
from io import BytesIO

def get_image_from_url(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        st.warning(f"Could not retrieve image from {url}")
        return None


def calculate_histogram(image):
    hist = np.zeros(256)
    for pixel in image.getdata():
        hist[pixel] += 1
    return hist

def cumulative_histogram(hist):
    cum_hist = hist.copy()
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]
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

def histogram_dashboard():
    st.title('Histogram Analysis')
    
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
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_container_width=True)
        col2.image(img_gray, caption="Grayscale", use_container_width=True)
        
        st.subheader("Image Histogram")
        hist = calculate_histogram(img_gray)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(256), hist)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Grayscale Histogram")
        st.pyplot(fig)
        
        st.subheader("Histogram Operations")
        operation = st.selectbox(
            "Select Operation", 
            ["Histogram Equalization", "Cumulative Histogram"]
        )
        
        if operation == "Histogram Equalization":
            equalized = histogram_equalization(img_gray)
            
            st.subheader("Histogram Equalization Result")
            col1, col2 = st.columns(2)
            col1.image(img_gray, caption="Original Grayscale", use_container_width=True)
            col2.image(equalized, caption="Equalized", use_container_width=True)
            
            eq_hist = calculate_histogram(equalized)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(256), eq_hist)
            ax.set_xlim(0, 255)
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Equalized Histogram")
            st.pyplot(fig)
            
        elif operation == "Cumulative Histogram":
            hist = calculate_histogram(img_gray)
            cum_hist = cumulative_histogram(hist)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(256), cum_hist)
            ax.set_xlim(0, 255)
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Cumulative Frequency")
            ax.set_title("Cumulative Histogram")
            st.pyplot(fig)
            
if __name__ == "__main__":
    histogram_dashboard()