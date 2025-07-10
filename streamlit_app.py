import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
#from streamlit_drawable_canvas import st_canvas
#from streamlit_image_annotation import detection
from streamlit_cropper import st_cropper
from PIL import Image

st.set_page_config(layout="wide")
st.title("Manga Translator")
stage = 0

# Stage 0: upload image
uploadedfile = st.file_uploader(label="Upload image", type=['jpg', 'png'], label_visibility = 'collapsed')
#uploadedfile = st.file_uploader(label = "You can choose up to 13 image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files = True, label_visibility = 'collapsed')

if uploadedfile is not None:
    st.write("Image successfully uploaded")
    #img = plt.imread(uploadedfile)
    #plt.axis('off')
    #plt.imshow(img);
    #st.image(img)
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
        conf = st.slider("Conf", 0.0, 1.0, (0.2))
    with col2:
        iou = st.slider("IOU", 0.0, 1.0, (0.7))  

    # get predictions with the model
    if st.button("Get bboxes", type="primary"):
        image_bytes = uploadedfile.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(source = orig_image, show = False, show_labels = False, save = False, conf = conf, iou = iou)
        stage = 2

# Stage 2: show and adjust bboxes
if stage > 1:
    realtime_update = True
    box_color = 'red'
    stroke_width = 3
    aspect_ratio = None
    return_type = 'box'

    if uploadedfile:
        img = Image.open(uploadedfile)
        rect = st_cropper(
            img,
            realtime_update = realtime_update,
            box_color = box_color,
            aspect_ratio = aspect_ratio,
            return_type = return_type,
            stroke_width = stroke_width
            )
        raw_image = np.asarray(img).astype('uint8')
        left, top, width, height = tuple(map(int, rect.values()))
        st.write(rect)
        masked_image = np.zeros(raw_image.shape, dtype='uint8')
        masked_image[top:top + height, left:left + width] = raw_image[top:top + height, left:left + width]
        st.image(Image.fromarray(masked_image), caption='masked image')