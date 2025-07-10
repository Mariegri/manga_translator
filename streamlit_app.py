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

# get image url
#from streamlit import config
#from streamlit.session_data import get_url
#from streamlit.elements.image import (
#    _BytesIO_to_bytes,
#    _normalize_to_bytes,
#    MAXIMUM_CONTENT_WIDTH,
##)
#from streamlit.in_memory_file_manager import (
#    _calculate_file_id,
#    _get_extension_for_mimetype,
#    STATIC_MEDIA_ENDPOINT,
#)

#def img_url(image):
#    mimetype = image.type
#    data = _BytesIO_to_bytes(image)
#    data, mimetype = _normalize_to_bytes(data, MAXIMUM_CONTENT_WIDTH, mimetype)
#    extension = _get_extension_for_mimetype(mimetype)
#    file_id = _calculate_file_id(data=data, mimetype=mimetype)
#    URL = get_url(config.get_option("browser.serverAddress"))
#    return "{}{}/{}{}".format(URL, STATIC_MEDIA_ENDPOINT, file_id, extension)


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










    #detection(
    #    image_path = img_url(uploadedfile),
    #    label_list = None, #(List[str])
    #    bboxes = None, #Optional[List[List[int, int, int, int]]] = None,
    #    labels = None, #Optional[List[int]] = None,
    #    height = 512,
    #    width = 512,
    #    line_width = 2,
    #    use_space = False,
    #    key = None
    #)





    #img = plt.imread(uploadedfile)
    #fig, ax = plt.subplots()
    #plt.axis('off')
    #plt.imshow(img);
    #st.image(img)

    #bboxes = res[0].boxes.xyxy
    #for bbox in bboxes:
    #    bbox = bbox.numpy()
        #st.write(bbox)
        #bbox = bbox.cpu().detach().numpy()
    #    rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 2, edgecolor = 'red', facecolor = 'none', lw = 2)
    #    ax.add_patch(rectangle)
                
    #plt.imshow(img)
    #plt.show();
    #st.image(img)
    #mpl.pyplot.close();



    #st.write(res[0].boxes.xyxy.numpy())

    #with st.echo("below"):
    #    drawing_mode = "rect"
    #    stroke_width = 2
    #    stroke_color = 'red'
    #    bg_image = uploadedfile
    #    realtime_update = True

        # Create a canvas component
    #    canvas_result = st_canvas(
    #        fill_color = None,  
    #        stroke_width = stroke_width,
    #        stroke_color = stroke_color,
    #        background_color = None,
    #        background_image = Image.open(bg_image) if bg_image else None,
    #        update_streamlit = realtime_update,
    #        height = 250,
    #        drawing_mode = drawing_mode,
    #        point_display_radius = 0,
    #        display_toolbar = True,
    #        key = "full_app",
    #    )

        # Do something interesting with the image data and paths
    #    if canvas_result.image_data is not None:
    #        st.image(canvas_result.image_data)
    #    if canvas_result.json_data is not None:
    #        objects = pd.json_normalize(canvas_result.json_data["objects"])
    #        for col in objects.select_dtypes(include=["object"]).columns:
    #            objects[col] = objects[col].astype("str")
    #        st.dataframe(objects)


