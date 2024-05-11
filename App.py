import streamlit as st
import os
from PIL import Image
import io
# import pickle
import numpy as np
import cv2
from utils import show_images, get_human, get_features_vector, yolo_draw_bounding_boxes, sort
from ultralytics import YOLO
from deepface import DeepFace
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
# from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
from transformers import AutoImageProcessor, Dinov2Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "facebook/dinov2-small"

session_state = st.session_state

if not hasattr(session_state, 'model'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.model = YOLO('yolov8l-face.pt')

if not hasattr(session_state, 'image_processor'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

if not hasattr(session_state, 'dino'):
    # Đọc mô hình chỉ một lần khi ứng dụng bắt đầu
    session_state.dino = Dinov2Model.from_pretrained("facebook/dinov2-small")

st.title('Image-based character retrieval from movies')

selected_option = st.selectbox('Select movie', ['Like Me', 'Calloused Hands', 'Liberty Kid', 'Losing Ground', 'Memphis'])

if (selected_option == 'Like Me'):
    film = 'like_me'
elif (selected_option == 'Calloused Hands'):
    film = 'Calloused_Hands'
elif (selected_option == 'Liberty Kid'):
    film = 'Liberty_Kid'
elif (selected_option == 'Losing Ground'):
    film = 'losing_ground'
elif (selected_option == 'Memphis'):
    film = 'Memphis'

facenet_persons = pd.read_csv("../input/new-index/like_me-persons-facenet.csv")
dino_persons = pd.read_csv("../input/new-index/like_me-persons-dino.csv")
ground_truth = set(pd.read_excel("../input/ground-truth/ground_truth/like_me/Kiya.xlsx")["Full"])

facenet_index = faiss.read_index("./indices/like_me-index-facenet.index")
dino_index = faiss.read_index("./indices/like_me-index-dino.index")
    
character = st.selectbox('Choose your desired character', os.listdir(f'./query/{film}/'))

k = st.slider('Select k (top most similar persons in movie)', 1, len(dino_persons), 1)

top_res = st.slider('Choose the number of shots to be displayed', 1, 200, 1)

st.write(f'Query images for {character} in {selected_option}')
query_images = []
for i in os.listdir(f'./query/{film}/{character}/'):
    img = cv2.imread(f'./query/{film}/{character}/{i}')
    img = cv2.resize(img, (100,100))
    query_images.append(img)
if(query_images != []):
    checkboxes = show_images(query_images)
    choices = st.multiselect('Choose query images:', checkboxes)

# method = st.selectbox('Feature extraction method', ['CLIP', 'DINOv2'])

query_button = st.button("Run the query")

if(query_button and choices):

    chose_images = []
    for choice in choices:
        tmp = choice.split(' ')
        tmp = int(tmp[1])
        chose_images.append(query_images[tmp-1])
    chose_images = np.array(chose_images)

facenet_query_features = []
dino_query_features = []

for img in chose_images:
    img = cv.imread(f"../input/movie-query/like_me/like_me/Kiya/{img_name}")
    frame_persons = yolo_draw_bounding_boxes(img)
    if frame_persons != []: 
        for person in frame_persons:
            embedding = DeepFace.represent(person, model_name="Facenet", enforce_detection=False)
            # tempt = np.reshape(i, (1, -1))
            embedding = np.array(embedding[0]["embedding"])
            print(embedding.shape)
            facenet_query_features.append(embedding)
    else:
        inputs = image_processor(img, return_tensors="pt")
        with torch.no_grad():
            embeddings = dino(**inputs).last_hidden_state
            embeddings = embeddings.mean(axis=1)
            vectors = embeddings.detach().cpu().numpy()
            vectors = np.float32(vectors)
            vectors = np.reshape(vectors, (1, -1))
           # faiss.normalize_L2(vectors)
            for i in vectors:
                tempt = np.reshape(i, (1, -1))
                print(i.shape)
                dino_query_features.append(i)
    facenet_query_features = np.array(facenet_query_features)
    facenet_query_features = facenet_query_features.astype("float32")
    # frame_features = np.reshape(facenet_frame_features, (frame_features.shape[0], frame_features.shape[1]*frame_features.shape[2]))
    faiss.normalize_L2(facenet_query_features)
    dino_query_features = np.array(dino_query_features)
    dino_query_features = dino_query_features.astype("float32")
    # frame_features = np.reshape(facenet_frame_features, (frame_features.shape[0], frame_features.shape[1]*frame_features.shape[2]))
    faiss.normalize_L2(dino_query_features)


    relevant_retrieved_shots_facenet = []
    relevant_retrieved_shots_dino = []
    retrieved_shots_facenet = OrderedSet()
    retrieved_shots_dino = OrderedSet()

    
    D1, I1 = facenet_index.search(facenet_query_features, k)
    search_results_facenet = OrderedSet()
    for query in I1:
        query_set = OrderedSet(query)
        search_results_facenet.update(query_set)
    if -1 in search_results_facenet:
        search_results_facenet.remove(-1)
    for i in search_results_facenet:
        retrieved_shots_facenet.add(facenet_persons["shot"][i])
    # retrieved_shots_facenet.append(retrieved_shots_facenet_at_k)    
    # relevant_retrieved_shots_facenet.append(retrieved_shots_facenet_at_k & ground_truth)


    D1, I1 = dino_index.search(dino_query_features, k)
    search_results_dino = OrderedSet()
    for query in I1:
        query_set = OrderedSet(query)
        search_results_dino.update(query_set)
    if -1 in search_results_dino:
        search_results_dino.remove(-1)
    for i in search_results_dino:
        retrieved_shots_dino.add(dino_persons["shot"][i])
    # retrieved_shots_dino.append(retrieved_shots_dino_at_k)    
    # relevant_retrieved_shots_dino.append(retrieved_shots_dino_at_k & ground_truth)

    retrieved_shots = retrieved_shots_dino
    retrieved_shots.update(retrieved_shots_facenet)
    print(retrieved_shots)
    st.title('Results:')
    if (len(retrieved_shots)<top_res):
        top_res = len(retrieved_shots)
    rows = top_res//5+1
    columns = st.columns(5)
    count = len(retrieved_shots)-1
    for row in range(rows):
        columns = st.columns(5)
        for column in columns:
            if(count==top_res):
                break
            tmp = retrieved_shots[count].split('-')
            shot = int((tmp[2].split('_'))[1])
            scene = int(tmp[1])
            column.video(f'./shots/{film}/{film}-{scene}/{film}-{scene}-shot_{shot}.webm')
            count-=1