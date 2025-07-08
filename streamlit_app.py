import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Manga Translator")

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = plt.imread(uploadFile)
    plt.axis('off')
    plt.imshow(img);
    st.image(img)
    #st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")