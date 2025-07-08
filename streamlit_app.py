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

# Stage 0: upload image
#uploadedfile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
uploadedfile = st.file_uploader(label = "You can choose up to 13 image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files = True)
if len(uploadedfile) > 13:
    st.write("You can choose only up to 13 images")
else:
    images_df = pd.DataFrame(columns = [uploadedfile[i] for i in range(len(uploadedfile))])



st.write(len(uploadedfile))

img = plt.imread(uploadedfile[2])
plt.axis('off')
plt.imshow(img);
st.image(img)

if uploadedfile is not None:
    img = plt.imread(uploadedfile)
    plt.axis('off')
    plt.imshow(img);
    st.image(img)
    stage = 1
else:
    st.write("Make sure you image is in JPG/PNG Format.")

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
    if st.button("Get text", type="primary"):
        image_bytes = uploadedfile.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(source = orig_image, show = False, show_labels = False, save = False, conf = conf, iou = iou)
        stage = 2

# Stage 2: show and adjust bboxes
if stage > 1:
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