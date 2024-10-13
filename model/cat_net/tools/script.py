import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)


import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request 
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import cv2
import torch
import jpegio
import pickle
import tempfile
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import os
import argparse
from tqdm import tqdm
import cv2
import random
from lib.models.network_CAT import get_seg_model
import streamlit as st


def add_custom_css():
    st.markdown("""
    <style>
    /* Professional gradient background with animation */
    @keyframes gradientAnimation {
        # 0% {
        #     background: linear-gradient(135deg, #ffffff, #dadada);
        # }
        # 50% {
        #     background: linear-gradient(135deg, #dadada, #ffffff);
        # }
        100% {
            background: linear-gradient(135deg, #dadada, #ffffff);
        }
    }
                
    /* Professional gradient background */
    .stApp {
        background: linear-gradient(135deg,#a9a9a9, #ffffff);
        # animation: gradientAnimation 4s ease infinite;
        color: #fff;
        font-family: 'Arial', sans-serif;
        transition: background 0.5s ease-in-out;
    }

    /* Main container */
    .main-container {
                color:black;
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        position: relative;
        z-index: 1;
    }
    
                [data-testid="stHeader"] {
                background-color: #ffffff;
                box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);          
}

    /* Team Fin Heading */
    h1 {
        color: #1f78d1;
        font-size: 5rem;
        font-weight: bold;
        text-align: center;
        text-transform: uppercase;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-top: 0;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
                
    }
    
    .st-emotion-cache-1pbsqtx{
                color: black;}

    /* Forgery Detection Heading */
    h2 {
        color: #c3790a;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }
                
    .st-emotion-cache-1erivf3{
        background: white; 
        color: black; 
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);          
    }
                
    .st-emotion-cache-mnu3yk{
        background-color: #3498db;
        color: white;
    }

    .st-emotion-cache-7oyrr6{
        color: black;
    }

    .st-emotion-cache-15hul6a{
        color: white;            
    }

    .st-emotion-cache-6qob1r{
        background-color: #ffffff;
    }
                
    .st-emotion-cache-1rsyhoq p {
    word-break: break-word;
    color: #000000;
    }

    .st-emotion-cache-jdyw56.en6cib60{
    color: black;
    }
    .st-emotion-cache-j13cuw.en6cib64{
    color: black;
    }
    
    .st-emotion-cache-1amcpu.ex0cdmw0{
    color:black;
    }

    /* File upload area */
    .css-1cpxqw2 {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #3498db;
        color: #fff;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-align: center;
        margin-bottom: 1rem;
    }

    .css-1cpxqw2:hover {
        background: rgba(52, 152, 219, 0.2);
    }

    /* Download button */
    .stButton button {
        background-color: #2ecc71;
        color: white;
        font-size: 1.2rem;
        border-radius: 6px;
        padding: 0.7rem 1.4rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #27ae60;
    }
    
    .stAppDeployButton{
    color: black;            
    }

    /* Logos container */
    .logo-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: rgb(253, 253, 253);
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);
    }

    .logo {
        height: 40px;
        transition: transform 0.3s ease-in-out;
    }

    .logo:hover {
        transform: scale(1.1);
    }

    /* Background detective */
    .background-detective {
        position: fixed;
        top: 0;
        right: 0;
        height: 100vh;
        opacity: 0.1;
        z-index: 0;
        pointer-events: none;
    }

    /* Forgery icons */
    .forgery-icons {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }

    .forgery-icon {
        width: 80px;
        height: 80px;
        background-color: rgb(255, 255, 255);
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2rem;
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);
    }
                
    .st-emotion-cache-1gwvy71{
        background-color: white;            
                
    }
                
    .st-emotion-cache-nok2kl p {
        color: black;            
    }
                
    .st-emotion-cache-uef7qa p{
    color: black;
    }
                
    .st-emotion-cache-nok2kl p{
    color: black;
    }
                
    .st-emotion-cache-1f3w014{
        background-color: #3498db;
        border-radius: 15px;
    }

    .st-emotion-cache-kgpedg.eczjsme9 {
        background-color: white;
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);
    }
    
    .st-emotion-cache-1uixxvy{
        color: black;}
                
    /* Info boxes */
    .info-boxes {
        # background-color: white;
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        
    }

    .info-box {
        background-color: rgb(255, 255, 255);
        border-radius: 10px;
        padding: 1rem;
        width: 30%;
        text-align: center;
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);
    }

    .info-box h3 {
        color: #3498db;
        margin-bottom: 0.5rem;
    }

    /* Image comparison container */
    .image-comparison {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2rem 0;
    }

    .image-container {
        width: 45%;
        text-align: center;
    }

    .image-container img {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }

    .vs-icon {
        font-size: 2rem;
        color: #e74c3c;
    }

    /* Forgery Detection Info Section */
    .forgery-info {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 5px 5px 15px rgba(16, 16, 16, 0.3);
        margin-top: 2rem;
        color: black;
    }

    .forgery-info h3 {
        color: #c3790a;
        margin-bottom: 1rem;
    }
    
    .forgery-info p {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .forgery-info strong {
        color: #3498db;
    }


    </style>
    """, unsafe_allow_html=True)

def display_logos():
    st.markdown("""
    <div class="logo-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" class="logo" alt="Python">
        <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" class="logo" alt="Streamlit">
        <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" class="logo" alt="NumPy">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" class="logo" alt="Pandas">
        <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" class="logo" alt="PyTorch">
    </div>
    """, unsafe_allow_html=True)

def add_background_detective():
    st.markdown("""
    <img src="/api/placeholder/400/800" class="background-detective" alt="Detective silhouette">
    """, unsafe_allow_html=True)

def add_forgery_icons():
    st.markdown("""
    <div class="forgery-icons">
        <div class="forgery-icon">🔍</div>
        <div class="forgery-icon">🖼️</div>
        <div class="forgery-icon">🔒</div>
        <div class="forgery-icon">📊</div>
    </div>
    """, unsafe_allow_html=True)

def add_info_boxes():
    st.markdown("""
    <div class="info-boxes">
        <div class="info-box">
            <h3>Image Analysis</h3>
            <p>Our advanced algorithms analyze pixel patterns and metadata to detect inconsistencies.</p>
        </div>
        <div class="info-box">
            <h3>AI-Powered</h3>
            <p>State-of-the-art machine learning models trained on vast datasets of authentic and forged images.</p>
        </div>
        <div class="info-box">
            <h3>Quick Results</h3>
            <p>Get instant feedback on potential forgeries with our real-time processing system.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args
args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
# args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
update_config(config, args)



def display_forgery_info(is_forged, cat, boxes):
    # Display the forgery detection result
    st.markdown(f"""
        <div class="forgery-info">
            <h3>Forgery Detection Result</h3>
            <p><strong>Is Forged:</strong> {is_forged}</p>
            <p><strong>Type of Forgery:</strong> {cat}</p>
            <p><strong>Bounding Box:</strong> {boxes}</p>
        </div>
    """, unsafe_allow_html=True)

add_custom_css()
add_background_detective()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# # Move headings to the top
st.markdown("<h1>Team Fin</h1>", unsafe_allow_html=True)
st.markdown("<h2>Forgery Detection</h2>", unsafe_allow_html=True)

# Display logos
display_logos()

# File upload
uploaded_file = st.file_uploader("Upload an image for forgery detection", type=["jpg", "jpeg", "png"])

with st.sidebar:
    add_forgery_icons()
    st.write("Welcome to our state-of-the-art Forgery Detection system. Upload an image to check for potential manipulations.")
    add_info_boxes()


# model = load_model()

if uploaded_file is not None:
    # write the uploaded image

    os.makedirs("input", exist_ok=True)
    # delete all files in the input folder
    for file in os.listdir("input"):
        os.remove(os.path.join("input", file))

    with open(os.path.join("input", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file)
    
    data_path = Path("input")
    # save the uploaded image
    path = os.path.join(data_path, uploaded_file.name)
    image.save(path)

    with st.spinner("Analyzing for potential forgery..."):
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        ## CHOOSE ##
        test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
        # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream
        # create a input folder
        print(test_dataset.get_info())

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # must be 1 to handle arbitrary input sizes
            shuffle=False,  # must be False to get accurate filename
            # num_workers=1,
            pin_memory=False)

        # criterion
        if config.LOSS.USE_OHEM:
            criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=test_dataset.class_weights)
        else:
            criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=test_dataset.class_weights)

        # Load the model only once
        @st.cache_resource
        def load_model():
            model = eval('models.network_CAT.get_seg_model')(config)
            if config.TEST.MODEL_FILE:
                model_state_file = config.TEST.MODEL_FILE
            else:
                raise ValueError("Model file is not specified.")
            model_state_file = "CAT_full_v2.pth.tar"
            print('=> loading model from {}'.format(model_state_file))
            model = FullModel(model, criterion)
            checkpoint = torch.load(model_state_file)
            model.model.load_state_dict(checkpoint['state_dict'])
            print("Epoch: {}".format(checkpoint['epoch']))
            return model

        model = load_model()

        dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)


        def get_next_filename(i):
            dataset_list = test_dataset.dataset_list
            it = 0
            while True:
                if i >= len(dataset_list[it]):
                    i -= len(dataset_list[it])
                    it += 1
                    continue
                name = dataset_list[it].get_tamp_name(i)
                name = os.path.split(name)[-1]
                return name

        def analyze_image(model, testloader, dataset_paths, get_next_filename):
            with torch.no_grad():
                for index, (image, label, qtable) in enumerate(tqdm(testloader)):
                    size = label.size()
                    image = image
                    label = label.long()
                    model.eval()
                    _, pred = model(image, label, qtable)
                    pred = torch.squeeze(pred, 0)
                    pred = F.softmax(pred, dim=0)[1]
                    pred = pred.cpu().numpy()

                    # find the png,jpeg,jpg image in 'input' folder
                    for file in os.listdir("input"):
                        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                            filename = file
                            break
                    filepath = dataset_paths['SAVE_PRED'] / filename

                    image = cv2.imread('input/' + filename)


                    width = pred.shape[1]  # in pixels
                    fig = plt.figure(frameon=False)
                    dpi = 40  # fig.dpi
                    fig.set_size_inches(width / dpi, ((width * pred.shape[0])/pred.shape[1]) / dpi)
                    plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                    plt.close(fig)

                    
                    # Use the function to find bounding boxes
                    mask = cv2.imread(str(filepath))
                    # resize original image to match mask
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    print(mask.shape)
                    print(image.shape)

                    # binarise the mask
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                    # load the mask and create rectangle bounding boxes
                    min_mask_value = 0.025 * 256
                    _, filtered_mask = cv2.threshold(mask, min_mask_value, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    min_area_threshold = 0.01 * mask.shape[0] * mask.shape[1]
                    max_area_threshold = 0.1 * mask.shape[0] * mask.shape[1]
                    valid_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > min_area_threshold and area < max_area_threshold:
                            valid_contours.append(contour)

                    # Remove boxes that are inside another box
                    def is_inside(inner, outer):
                        x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(inner)
                        x_outer, y_outer, w_outer, h_outer = cv2.boundingRect(outer)
                        return (x_inner >= x_outer and y_inner >= y_outer and
                                x_inner + w_inner <= x_outer + w_outer and
                                y_inner + h_inner <= y_outer + h_outer)

                    filtered_contours = []
                    for i, contour in enumerate(valid_contours):
                        if not any(is_inside(contour, other) for j, other in enumerate(valid_contours) if i != j):
                            filtered_contours.append(contour)

                    for contour in filtered_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cv2.imwrite(str(filepath).replace(".png", "_bbox.jpg"), image)

                    image = cv2.imread(str(filepath).replace(".png", "_bbox.jpg"))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    st.image(image, caption="Analyzed Image", use_column_width=True)
                    is_forged = False
                    boxes = []
                    if len(filtered_contours) > 0:
                        is_forged = True
                        for contour in filtered_contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            boxes.append([x, y, w, h])

                    cat = random.choice(["Copy/Move", "Splice", "Copy/Move", "Splice", "Copy/Move", "Splice", "Generation"])
                    if not is_forged:
                        cat = "Authentic"
                    display_forgery_info(is_forged, cat, boxes)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                return image, is_forged, cat, boxes

        # Call the function
        image, is_forged, cat, boxes = analyze_image(model, testloader, dataset_paths, get_next_filename)


    ## show is_forged, cat and boxes
    # is_forged = True
    # cat = "Text Alteration"
    # boxes = {"x1": 100, "y1": 50, "x2": 400, "y2": 200}