import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def read_grayscale_image(image):
    return image.convert('L')

def image_to_2d(img_data, width):
    return [img_data[i:i+width] for i in range(0, len(img_data), width)]

def bitwise_not(image):
    return [[255 - pixel for pixel in row] for row in image]

def bitwise_and(image1, image2):
    return [[p1 & p2 for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def bitwise_or(image1, image2):
    return [[p1 | p2 for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def bitwise_xor(image1, image2):
    return [[p1 ^ p2 for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def bitwise_and_not(image1, image2):
    not_image2 = bitwise_not(image2)
    return bitwise_and(image1, not_image2)

def process_images(img1, img2):
    width, height = img1.size
    img2 = img2.resize((width, height))
    img1_gray = read_grayscale_image(img1)
    img2_gray = read_grayscale_image(img2)
    img1_data = list(img1_gray.getdata())
    img2_data = list(img2_gray.getdata())
    img1_2d = image_to_2d(img1_data, width)
    img2_2d = image_to_2d(img2_data, width)
    processed_images = {
        'G1 Original': np.array(img1),
        'G2 Original': np.array(img2),
        'Grayscale G1': np.array(img1_gray),
        'Grayscale G2': np.array(img2_gray),
        'NOT(G1)': np.array(bitwise_not(img1_2d)),
        'G1 AND G2': np.array(bitwise_and(img1_2d, img2_2d)),
        'G1 OR G2': np.array(bitwise_or(img1_2d, img2_2d)),
        'G1 AND NOT(G2)': np.array(bitwise_and_not(img1_2d, img2_2d)),
        'G1 XOR G2': np.array(bitwise_xor(img1_2d, img2_2d))
    }
    return processed_images

def get_image_from_url(url):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]

    def fetch_image(img_url):
        for agent in user_agents:
            try:
                response = requests.get(
                    img_url, 
                    headers={"User-Agent": agent},
                    timeout=10,
                    verify=True
                )
                
                content_type = response.headers.get('Content-Type', '').lower()
                
                valid_image_types = [
                    'image/jpeg', 'image/png', 'image/gif', 
                    'image/bmp', 'image/webp', 'image/svg+xml'
                ]
                
                if any(img_type in content_type for img_type in valid_image_types):
                    return Image.open(BytesIO(response.content))
                
            except (requests.RequestException, requests.Timeout) as e:
                st.warning(f"Error fetching {img_url}: {e}")
                continue
        
        return None

    if not url or not url.strip():
        st.warning("URL kosong")
        return None

    if not re.match(r'^https?://', url):
        url = 'http://' + url

    try:
        direct_image = fetch_image(url)
        if direct_image:
            return direct_image

        try:
            response = requests.get(url, headers={"User-Agent": user_agents[0]}, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            img_tags = soup.find_all("img")
            
            for img_tag in img_tags:
                img_src = img_tag.get('src') or img_tag.get('data-src')
                if not img_src:
                    continue

                if not re.match(r'^https?://', img_src):
                    img_src = urljoin(url, img_src)

                web_image = fetch_image(img_src)
                if web_image:
                    return web_image

        except Exception as e:
            st.warning(f"Error parsing webpage: {e}")

        parsed_url = urlparse(url)
        alternative_urls = [
            f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico",
            f"{parsed_url.scheme}://{parsed_url.netloc}/logo.png",
            f"{parsed_url.scheme}://{parsed_url.netloc}/icon.png"
        ]

        for alt_url in alternative_urls:
            alt_image = fetch_image(alt_url)
            if alt_image:
                return alt_image

    except Exception as e:
        st.warning(f"Unexpected error: {e}")

    st.warning(f"Could not retrieve image from {url}")
    return None

def main():
    st.set_page_config(page_title="PCD Kelompok 2 Tugas 1", layout="wide")
    st.title('Image Processing')
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_img1 = st.file_uploader("Upload First Image", type=['jpg', 'png', 'jpeg'])
        img1_url = None if uploaded_img1 else st.text_input("Or enter image URL for First Image")
        
        if uploaded_img1:
            img1 = Image.open(uploaded_img1)
        elif img1_url:
            img1 = get_image_from_url(img1_url)
        else:
            img1 = None
        
        if img1:
            st.image(img1, caption="First Image Preview", use_column_width=True)
    
    with col2:
        uploaded_img2 = st.file_uploader("Upload Second Image", type=['jpg', 'png', 'jpeg'])
        img2_url = None if uploaded_img2 else st.text_input("Or enter image URL for Second Image")
        
        if uploaded_img2:
            img2 = Image.open(uploaded_img2)
        elif img2_url:
            img2 = get_image_from_url(img2_url)
        else:
            img2 = None
        
        if img2:
            st.image(img2, caption="Second Image Preview", use_column_width=True)
    
    st.sidebar.header('Image Processing Filters')
    filter_options = [
        'G1 Original', 'G2 Original', 
        'Grayscale G1', 'Grayscale G2', 
        'NOT(G1)', 'G1 AND G2', 
        'G1 OR G2', 'G1 AND NOT(G2)', 
        'G1 XOR G2'
    ]
    selected_filters = st.sidebar.multiselect("Select Filters", filter_options, default=filter_options)
    
    if img1 and img2:
        processed_images = process_images(img1, img2)
        
        if selected_filters:
            st.header('Image Processing Results')
            result_cols = st.columns(3)
            col_index = 0
            for option in selected_filters:
                with result_cols[col_index]:
                    st.image(processed_images[option], caption=option, use_column_width=True)
                col_index = (col_index + 1) % 3
    else:
        st.warning('Please upload two images to process or enter image URLs')

if __name__ == '__main__':
    main()
