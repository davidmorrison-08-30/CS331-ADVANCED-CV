import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import scenedetect
from utils import detect_scenes, get_human, get_features_vector
from ultralytics import YOLO
import cv2
import imutils
import math
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from keras.applications.vgg16 import preprocess_input
from ultralytics.utils.plotting import Annotator
import time
import random
import openai
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss

model_name = "openai/clip-vit-base-patch16"

# Khai báo biến session
session_state = st.session_state

# Kiểm tra xem mô hình đã được đọc hay chưa
if not hasattr(session_state, 'processor'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.processor = CLIPProcessor.from_pretrained(model_name)

if not hasattr(session_state, 'clip_model'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.clip_model = CLIPProcessor.from_pretrained(model_name)

if not hasattr(session_state, 'model'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.model = YOLO('yolov8l.pt')

st.title('Truy Vấn Ảnh Trong Một Video Sử Dụng Clip và YoloV8')

k = st.slider('Cosine Threshold',0, 100,30)

uploaded_files = st.file_uploader("Upload query images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        

video_file = st.file_uploader("Upload query video", type=["mp4", "webm"])
query_button = st.button("Query")
if video_file and query_button and uploaded_files:
    # st.video(video_file)
    # Get Query Features
    st.title('Relevant Scene:')
    query_list = list()
    for i in uploaded_files:
        # inp = cv2.imread(i)
        pil_image = Image.open(i)
        image_np = np.array(pil_image)
        inp = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        res = get_human(session_state.model,inp,0.5)
        query_list.append(res)
    images = list()
    for i in query_list:
        for j in i:
            images.append(j)
    query_list = images
    
    # Get query features vector
    query_features = get_features_vector(query_list, session_state.clip_model, session_state.processor)
    
    faiss.normalize_L2(query_features)
    index = faiss.index_factory(query_features.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.ntotal
    index.add(query_features)
    
        
    scenes = detect_scenes(video_file)
    st.write('Query Features Done!!!')
    res_count = 0
    for scene in scenes:
        start_frame = time.time()
        scene_start_frame, scene_end_frame = scene[0], scene[1]
        video_capturer = cv2.VideoCapture(video_file)

        # Đặt vị trí của video capturer đến frame bắt đầu của cảnh
        video_capturer.set(cv2.CAP_PROP_POS_FRAMES, scene_start_frame)

        # Đọc từ frame bắt đầu đến frame kết thúc của cảnh
        for frame_num in range(scene_start_frame, scene_end_frame):
            flag = 0
            ret, frame = video_capturer.read()
            if not ret:
                break
            # st.video(video_capturer)
            objects = get_human(session_state.model,frame,0.5)
            tmp = list()
            for i in objects:
                tmp.append(i)
            objects = tmp
            image_features = get_features_vector(objects, session_state.clip_model, session_state.processor)
            k = 1
            if (image_features != None):
                D, I = index.search(image_features, k)
                similarity_values = 1 / (1 + D)
                for i,value in enumerate(similarity_values):
                    if (value[0]*100>k):
                        st.write(tmp+f' Score: {value[0]} Time: {time.time()-start_frame}')
                        st.video(video_capturer)
                        flag = 1
                        res_count += 1
                        break
            if (flag == 1):
                break
        video_capturer.release()
    if (res_count == 0):
        st.write('No relevants found!!!')