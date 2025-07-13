import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import plotly.express as px
from PIL import Image
from manga_ocr import MangaOcr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def get_translation(orig_image, bboxes):
    mocr = MangaOcr() 

    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    source_lang = 'jpn_Jpan'
    target_lang = "rus_Cyrl"
    translator = pipeline('translation', model = model, tokenizer = tokenizer, src_lang = source_lang, tgt_lang = target_lang, max_length = 400)
    image = Image.fromarray(image)   
    translations = []
    
    for bbox in bboxes:
        # get original texts
        cropped_img = orig_image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        text = mocr(cropped_img)
        #texts.append(text)

        # get translations
        output = translator(text)
        translated_text = output[0]['translation_text']
        translations.append(translated_text)
    return translations 


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
        conf = st.slider("Conf", 0.0, 1.0, (0.45))
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
    bboxes = bboxes.cpu().detach().numpy()
    @st.cache_data
    def show_bboxes(orig_image, bboxes):
        # add to figure
        fig, ax = plt.subplots(figsize = (15, 10))
        plt.imshow(orig_image);
        plt.axis('off')

        # add bboxes from model predictions
        for i in range(len(bboxes)):   
            bbox = bboxes[i]
            rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 2, edgecolor = 'red', facecolor = 'none', lw = 2)
            ax.add_patch(rectangle)
            bbox_name = 'bbox' + str(i)
            ax.annotate(bbox_name, xy = (bbox[0], bbox[1]), color = 'red')
                
        st.pyplot(fig)
    show_bboxes(orig_image, bboxes)
    stage = 3

if stage > 2:
    st.subheader("Translation")
        
    if st.button("Translate", type = "primary"):
        # show original and translations  
        col1, col2 = st.columns(2)
        with col1:
            show_bboxes(orig_image, bboxes)
        with col2:
            translations = get_translation(orig_image, bboxes)
            fig, ax = plt.subplots(figsize = (15, 10))
            plt.imshow(orig_image);
            plt.axis('off')

            # add translations
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                rectangle = mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 0, edgecolor = 'none', facecolor = 'white', lw = 0)
                ax.add_patch(rectangle)
                full_text = translations[i].split()
                text = ''
                for word in full_text:
                    text += word + '\n ' 
                    ax.text(((bbox[2] + bbox[0]) / 2), ((bbox[3] + bbox[1]) / 2), text, horizontalalignment = 'center', verticalalignment = 'center', fontsize = 8)

            st.pyplot(fig)

        