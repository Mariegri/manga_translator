import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Manga Translator")

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
#img_files = st.file_uploader(label="Choose an image files",
                 #type=['png', 'jpg', 'jpeg'],
                 #accept_multiple_files=True)

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


st.subheader("Find text")

#model.load_state_dict(torch.load('data/best.pt'))
model = YOLO('data/best.pt')

col1, col2 = st.columns(2)

with col1:
    conf = st.slider("Conf", 0.0, 1.0, (0.2))

with col2:
    iou = st.slider("IOU", 0.0, 1.0, (0.7))

res = model.predict(source = uploadFile, show = False, show_labels = False, save = False, conf = conf, iou = iou)

#im0 = run(source=open_cv_image, \
#  conf_thres=0.25, weights="runs/detect/yolov7.pt")