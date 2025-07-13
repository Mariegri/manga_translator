import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import plotly.express as px
from PIL import Image

st.set_page_config(layout="wide")
st.title("Manga Translator")
stage = 0

# Stage 0: upload image
uploadedfile = st.file_uploader(label="Upload image", type=['jpg', 'png'], label_visibility = 'collapsed')

if uploadedfile is not None:
    st.write("Image successfully uploaded")
    stage = 1
else:
    st.write("Make sure you image is in JPG/PNG Format")

# Stage 1: get bboxes using pretrained yolo-model
if stage > 0:
    st.subheader("Find text")
    model = YOLO('data/best.pt')

    # add conf and iou changers
    col1, col2 = st.columns(2)
    with col1:
        conf = st.slider("Conf", 0.0, 1.0, (0.4))
    with col2:
        iou = st.slider("IOU", 0.0, 1.0, (0.4))  

    # get predictions with the model
    if st.button("Get bboxes", type="primary"):
        image_bytes = uploadedfile.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(source = orig_image, show = False, show_labels = False, save = False, conf = conf, iou = iou)
        bboxes = res[0].boxes.xyxy
        stage = 2
        
# Stage 2: show and adjust bboxes
if stage > 1:   
    
    # add to figure
    fig, ax = plt.subplots(figsize = (15, 10))
    plt.imshow(orig_image);
    plt.axis('off')

    # add bboxes from model predictions
    for i in range(len(bboxes)):
        bbox = bboxes[i].cpu().detach().numpy()
        rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 2, edgecolor = 'red', facecolor = 'none', lw = 2)
        ax.add_patch(rectangle)
        bbox_name = 'bbox' + str(i)
        ax.annotate(bbox_name, xy = (bbox[0], bbox[1]), color = 'red')
            
    st.pyplot(fig)
    stage = 3

if stage > 2:
    st.subheader("Clean text")
    
    if st.button("Clean text", type = "primary"):
        img_h, img_w, can = orig_image.shape
        mask = np.zeros(orig_image.shape[:2], dtype="uint8")
        for bbox in bboxes:
            cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 255, thickness = -1)
        cleared_image = cv2.inpaint(orig_image, mask, 1, cv2.INPAINT_TELEA)
        st.write(cleared_image)

    #fig, ax = plt.subplots(figsize = (15, 10))
    #plt.imshow(cleared_image);
    #plt.axis('off')
    #st.pyplot(fig)