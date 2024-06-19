import streamlit as st
import os
from PIL import Image
import io
import pickle
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import cv2
import imutils
import math
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
import time
import random
# import openai
from transformers import CLIPProcessor, CLIPModel
# import torch
import faiss
from transformers import AutoImageProcessor, Dinov2Model
import re

def show_images(images):
    checkbox = []
    rows = len(images)//5+1
    columns = st.columns(5)
    count = 0
    for row in range(rows):
        columns = st.columns(5)
        for column in columns:
            if(count==len(images)):
                break
            column.image(images[count], caption=f"Image {count+1}", use_column_width=True)
            checkbox.append(f'Image {count+1}')
            count+=1
    return checkbox

def get_human(model, img, thres=0.5):
    res_list = list()
    try:
        os.mkdir(path)
    except:
        _=0
    img = img.copy()
    results = model.predict(img, verbose=False, conf=thres)
    for r in results:
        # annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            b = b.cpu().numpy().astype(int)
            top, left, bottom, right = b
            # print(f'bounding box: {b}')
            c = box.cls
            cropped_box = img[left:right, top:bottom]
            if (model.names[int(c)]=='person'):
                # cv2.imshow('abc', cropped_box)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                res_list.append(cropped_box)
                # cv2.imwrite(path+f'/{name}',cropped_box)
            # annotator.box_label(b, model.names[int(c)])
    # res = annotator.result()
    return res_list

def get_features_vector(imgs, model, processor):
    if(len(imgs)>0):
        inputs = processor(text=None, images=imgs, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features
    return None

def yolo_draw_bounding_boxes(image, yolo_model):
    lst_of_persons = []
    results = yolo_model.predict(image, verbose=False, conf=0.5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            b = b.cpu().numpy().astype(int)
            top, left, bottom, right = b
            # print(f'bounding box: {b}')
            c = box.cls
            person_img = image[left:right, top:bottom]
            label = yolo_model.names[int(c)]
            if label == "face":
                # label = yolo_model.names[int(c)]
                # print(coordinates)
                lst_of_persons.append(person_img)
    return lst_of_persons

def sort(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
