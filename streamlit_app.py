import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Manga Translator")
stage = 0

# Uploading the File to the Page
uploadedfile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
#img_files = st.file_uploader(label="Choose an image files",
                 #type=['png', 'jpg', 'jpeg'],
                 #accept_multiple_files=True)

# Checking the Format of the page

if uploadedfile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = plt.imread(uploadedfile)
    plt.axis('off')
    plt.imshow(img);
    st.image(img)
    #st.write("Image Uploaded Successfully")
    stage = 1
else:
    st.write("Make sure you image is in JPG/PNG Format.")

if stage > 0:
    st.subheader("Find text")

    #model.load_state_dict(torch.load('data/best.pt'))
    model = YOLO('data/best.pt')

    col1, col2 = st.columns(2)

    with col1:
        conf = st.slider("Conf", 0.0, 1.0, (0.2))

    with col2:
        iou = st.slider("IOU", 0.0, 1.0, (0.7))

    if uploadedfile is not None:
        # Read image file
        image_bytes = uploadedfile.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if st.button("Get text", type="primary"):
        res = model.predict(source = orig_image, show = False, show_labels = False, save = False, conf = conf, iou = iou)

    st.write(res[0].boxes)

'''
    # show yolo predictions with bboxes
    def show_pics(image_path, bboxes):
            
        # read image
        img = cv2.imread(image_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, can = img.shape
        plt.figure(figsize = (20, 15))
        fig, ax = plt.subplots()
        plt.axis('off')

        # add bboxes
        for bbox in bboxes:
            bbox = bbox.cpu().detach().numpy()
            rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'red', facecolor = 'none', lw = 2)
            ax.add_patch(rectangle)
                
        plt.imshow(img)
        plt.show();
        mpl.pyplot.close();
'''