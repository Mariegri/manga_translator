import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Manga Translator")
stage = 0

# Stage 0: upload image
uploadedfile = st.file_uploader(label="Upload image", type=['jpg', 'png'], label_visibility = 'collapsed')
#uploadedfile = st.file_uploader(label = "You can choose up to 13 image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files = True, label_visibility = 'collapsed')

#if uploadedfile is not None:
#    if len(uploadedfile) > 13:
#        st.write("You can choose only up to 13 images")
#        uploadedfile = []
#    else:
 #       'Images uploaded successfully'

#else:
#    st.write("Make sure you images are in JPG/PNG Format.")


#st.write(len(uploadedfile))
#st.dataframe(images_df)


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

    #images_df = pd.DataFrame(columns = [i for i in range(len(uploadedfile))])
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
    img = plt.imread(uploadedfile)
    fig, ax = plt.subplots()
    plt.axis('off')
    #plt.imshow(img);
    #st.image(img)

    bboxes = res[0].boxes.xyxy
    for bbox in bboxes:
        bbox = bbox.numpy()
        #st.write(bbox)
        #bbox = bbox.cpu().detach().numpy()
        rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 2, edgecolor = 'red', facecolor = 'none', lw = 2)
        ax.add_patch(rectangle)
                
    plt.imshow(img)
    plt.show();
    st.image(img)
    #mpl.pyplot.close();



    #st.write(res[0].boxes.xyxy.numpy())

