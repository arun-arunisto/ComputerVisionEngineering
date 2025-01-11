import cv2
from cv2 import dnn
import numpy as np
import streamlit as st
from PIL import Image
from converting_bnw_color import ConvertingBlackAndWhiteImagesToColoredImages
import time


st.set_page_config(page_title="Image Colorization")


st.write("## Turn Your Memories into :rainbow[Color]")

st.write(":camera_with_flash: Upload the black and white image you want to colorize")

st.sidebar.write("## Arun Arunisto :male-technologist:")
st.sidebar.write("Converting black and white images into color images using OpenCV and Deep Learning")
st.sidebar.write("---")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image :gear: \n", type=["png", "jpg", "jpeg"])

st.sidebar.write("---")
st.sidebar.write("## Contact Information:")
st.sidebar.write(":iphone: +91 8137856143")
st.sidebar.write(":email: arun.arunisto2@gmail.com")

st.sidebar.write("---")
st.sidebar.write("## Social Profiles:")
st.sidebar.write("[GitHub](https://github.com/arun-arunisto) | [LinkedIn](https://www.linkedin.com/in/arun-arunisto-685123171/) | [Instagram](https://www.instagram.com/arunisto/)")

bnwconverter = ConvertingBlackAndWhiteImagesToColoredImages()

def fixed_image(image):
    img = Image.open(image)
    col1.image(img, use_container_width=True)
    #adding a spinner while processing
    # Create a placeholder in col2 for the loader
    with col2:
        placeholder = st.empty()  # Placeholder for loader or image
        with placeholder.container():
            st.markdown("### Colorizing Image.... :hourglass:")  # Text for the loader
    #converting image into numpy array
    img_np = np.array(img)
    #converting image into color
    colorized = bnwconverter.colorize(img_np)
    time.sleep(3)
    placeholder.image(colorized[:, :, ::-1], use_container_width=True)

if my_upload is not None:
    fixed_image(my_upload)
else:
    st.write("## :point_left: Please Upload an Image :camera_with_flash:")
