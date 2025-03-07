import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import math


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

def bitwise_xnor(image1, image2):
    xor_result = bitwise_xor(image1, image2)
    return bitwise_not(xor_result)

def bitwise_nor(image1, image2):
    or_result = bitwise_or(image1, image2)
    return bitwise_not(or_result)

def bitwise_nand(image1, image2):
    and_result = bitwise_and(image1, image2)
    return bitwise_not(and_result)

def bitwise_and_not(image1, image2):
    not_image2 = bitwise_not(image2)
    return bitwise_and(image1, not_image2)

def bitwise_and_not_reverse(image1, image2):
    not_image1 = bitwise_not(image1)
    return bitwise_and(image2, not_image1)

def add(image1, image2):
    return [[min(p1 + p2, 255) for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def subtract(image1, image2):
    return [[max(p1 - p2, 0) for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def multiply(image1, image2):
    return [[min(p1 * p2 // 255, 255) for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def divide(image1, image2):
    return [[min(p1 // (p2 + 1), 255) for p1, p2 in zip(row1, row2)] for row1, row2 in zip(image1, image2)]

def logarithm(image):
    return [[min(int(math.log1p(pixel) * 45), 255) for pixel in row] for row in image]

def exponential(image):
    return [[min(int(math.exp(pixel / 255) * 255), 255) for pixel in row] for row in image]

def square_root(image):
    return [[int(math.sqrt(pixel) * 16) for pixel in row] for row in image]

def sine(image):
    return [[int((math.sin(pixel / 255 * 2 * math.pi) + 1) * 127.5) for pixel in row] for row in image]

def cosine(image):
    return [[int((math.cos(pixel / 255 * 2 * math.pi) + 1) * 127.5) for pixel in row] for row in image]

def tangent(image):
    return [[min(int((math.tan(pixel / 255 * math.pi) + 1) * 127.5), 255) for pixel in row] for row in image]

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def flip_horizontal(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def scale_image(image, scale_factor):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    return image.resize((new_width, new_height))

def translate_image(image, x_offset, y_offset):
    translated = Image.new("L", (image.width, image.height))
    translated.paste(image, (x_offset, y_offset))
    return translated


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
        'NOT(G2)': np.array(bitwise_not(img2_2d)),
        'G1 AND G2': np.array(bitwise_and(img1_2d, img2_2d)),
        'G1 OR G2': np.array(bitwise_or(img1_2d, img2_2d)),
        'G1 AND NOT(G2)': np.array(bitwise_and_not(img1_2d, img2_2d)),
        'G2 AND NOT(G1)': np.array(bitwise_and_not_reverse(img1_2d, img2_2d)),
        'G1 XOR G2': np.array(bitwise_xor(img1_2d, img2_2d)),
        'XNOR/equivalence': np.array(bitwise_xnor(img1_2d, img2_2d)),
        'NOR (NOT(G1 OR G2))': np.array(bitwise_nor(img1_2d, img2_2d)),
        'NAND (NOT(G1 AND G2))': np.array(bitwise_nand(img1_2d, img2_2d)),
        'G1 + G2': np.array(add(img1_2d, img2_2d)),
        'G1 - G2': np.array(subtract(img1_2d, img2_2d)),
        'G1 * G2': np.array(multiply(img1_2d, img2_2d)),
        'G1 / G2': np.array(divide(img1_2d, img2_2d)),
        'Log(G1)': np.array(logarithm(img1_2d)),
        'Log(G2)': np.array(logarithm(img2_2d)),
        'Exp(G1)': np.array(exponential(img1_2d)),
        'Exp(G2)': np.array(exponential(img2_2d)),
        '√(G1)': np.array(square_root(img1_2d)),
        '√(G2)': np.array(square_root(img2_2d)),
        'sin(G1)': np.array(sine(img1_2d)),
        'sin(G2)': np.array(sine(img2_2d)),
        'cos(G1)': np.array(cosine(img1_2d)),
        'cos(G2)': np.array(cosine(img2_2d)),
        'tan(G1)': np.array(tangent(img1_2d)),
        'tan(G2)': np.array(tangent(img2_2d)),
        'Rotate G1 (45°)': np.array(rotate_image(img1_gray, 45)),
        'Rotate G2 (45°)': np.array(rotate_image(img2_gray, 45)),
        'Flip G1 Horizontal': np.array(flip_horizontal(img1_gray)),
        'Flip G2 Horizontal': np.array(flip_horizontal(img2_gray)),
        'Flip G1 Vertical': np.array(flip_vertical(img1_gray)),
        'Flip G2 Vertical': np.array(flip_vertical(img2_gray)),
        'Scale G1 (0.5x)': np.array(scale_image(img1_gray, 0.5)),
        'Scale G2 (0.5x)': np.array(scale_image(img2_gray, 0.5)),
        'Translasi G1 (30, 30)': np.array(translate_image(img1_gray, 30, 30)),
        'Translasi G2 (30, 30)': np.array(translate_image(img2_gray, 30, 30)),
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

    with col2:
        uploaded_img2 = st.file_uploader("Upload Second Image", type=['jpg', 'png', 'jpeg'])
        img2_url = None if uploaded_img2 else st.text_input("Or enter image URL for Second Image")

        if uploaded_img2:
            img2 = Image.open(uploaded_img2)
        elif img2_url:
            img2 = get_image_from_url(img2_url)
        else:
            img2 = None

    st.sidebar.header('Image Processing Filters')

    logic_operations = [
        'NOT(G1)', 'NOT(G2)', 'G1 AND G2', 'G1 OR G2', 'G1 AND NOT(G2)', 'G2 AND NOT(G1)',
        'G1 XOR G2', 'XNOR/equivalence', 'NOR (NOT(G1 OR G2))', 'NAND (NOT(G1 AND G2))'
    ]

    arithmetic_operations = [
        'G1 + G2', 'G1 - G2', 'G1 * G2', 'G1 / G2',
        'Log(G1)', 'Log(G2)', 'Exp(G1)', 'Exp(G2)', '√(G1)', '√(G2)',
        'sin(G1)', 'sin(G2)', 'cos(G1)', 'cos(G2)', 'tan(G1)', 'tan(G2)'
    ]

    geometric_transformations = [
        'Rotate G1 (45°)', 'Rotate G2 (45°)', 'Flip G1 Horizontal', 'Flip G2 Horizontal',
        'Flip G1 Vertical', 'Flip G2 Vertical', 'Scale G1 (0.5x)', 'Scale G2 (0.5x)',
        'Translasi G1 (30, 30)', 'Translasi G2 (30, 30)'
    ]

    selected_logic = st.sidebar.multiselect("Logic Operations", logic_operations)
    selected_arithmetic = st.sidebar.multiselect("Arithmetic Operations", arithmetic_operations)
    selected_geometry = st.sidebar.multiselect("Geometric Transformations", geometric_transformations)

    selected_filters = selected_logic + selected_arithmetic + selected_geometry

    if img1 and img2:
        processed_images = process_images(img1, img2)
        
        st.header('Original & Grayscale Images')
        result_cols = st.columns(4)
        result_cols[0].image(img1, caption='Original G1', use_container_width=True)
        result_cols[1].image(img2, caption='Original G2', use_container_width=True)
        result_cols[2].image(img1.convert('L'), caption='Grayscale G1', use_container_width=True)
        result_cols[3].image(img2.convert('L'), caption='Grayscale G2', use_container_width=True)

        if selected_filters:
            st.header('Image Processing Results')
            result_cols = st.columns(3)
            col_index = 0

            for option in selected_filters:
                if option in processed_images and processed_images[option] is not None:
                    processed_images[option] = np.clip(processed_images[option], 0, 255).astype(np.uint8)
                    image_pil = Image.fromarray(processed_images[option])

                    with result_cols[col_index]:
                        st.image(image_pil, caption=option, use_container_width=True)

                    col_index = (col_index + 1) % 3
                else:
                    st.warning(f"Gambar untuk opsi '{option}' tidak ditemukan atau belum diproses.")

if __name__ == '__main__':
    main()