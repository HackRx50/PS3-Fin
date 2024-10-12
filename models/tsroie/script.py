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
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, 'models'))
sys.path.append(os.path.join(current_path, 'models/tsroie'))
sys.path.append("E:\Bajaj\DocTamper\models")
# print(current_path)
from swins import *
from dtd import seg_dtd
# import torch.nn as nn

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
new_qtb = np.array([[ 2,  1,  1,  2,  2,  4,  5,  6],[ 1,  1,  1,  2,  3,  6,  6,  6],[ 1,  1,  2,  2,  4,  6,  7,  6],[ 1,  2,  2,  3,  5,  9,  8,  6],[ 2,  2,  4,  6,  7, 11, 10,  8],[ 2,  4,  6,  6,  8, 10, 11,  9],[ 5,  6,  8,  9, 10, 12, 12, 10],[ 7,  9, 10, 10, 11, 10, 10, 10]],dtype=np.int32).reshape(64,).tolist()



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
        background: linear-gradient(135deg,#dadada, #ffffff);
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
        color: #3498db;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
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
        background-color: #cfcdca;
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

def add_image_comparison():
    st.markdown("""
    <div class="image-comparison">
        <div class="image-container">
            <img src="/api/placeholder/400/300" alt="Original Image">
            <p>Original Image</p>
        </div>
        <div class="vs-icon">VS</div>
        <div class="image-container">
            <img src="/api/placeholder/400/300" alt="Forged Image">
            <p>Potential Forgery</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_model():
    if 'model' not in st.session_state:
        with st.spinner("Initializing model..."):
            model = seg_dtd("", 2).to(device)
            model = nn.DataParallel(model)
            loader = torch.load('pths/dtd_sroie.pth', map_location='cpu')['state_dict']
            model.load_state_dict(loader)
            st.session_state.model = model
    return st.session_state.model

def crop_img(img, jpg_dct, crop_size=512, mask=None):
    if mask is None:
        use_mask=False
    else:
        use_mask=True
        crop_masks = []

    h, w, c = img.shape
    h_grids = h // crop_size
    w_grids = w // crop_size

    crop_imgs = []
    crop_jpe_dcts = []

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_img = img[y1:y2, x1:x2, :]
            crop_imgs.append(crop_img)
            crop_jpe_dct = jpg_dct[y1:y2, x1:x2]
            crop_jpe_dcts.append(crop_jpe_dct)
            if use_mask:
                if mask[y1:y2, x1:x2].max()!=0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w%crop_size!=0:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_imgs.append(img[y1:y2,w-512:w,:])
            crop_jpe_dcts.append(jpg_dct[y1:y2,w-512:w])
            if use_mask:
                if mask[y1:y2,w-512:w].max()!=0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if h%crop_size!=0:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            crop_imgs.append(img[h-512:h,x1:x2,:])
            crop_jpe_dcts.append(jpg_dct[h-512:h,x1:x2])
            if use_mask:
                if mask[h-512:h,x1:x2].max()!=0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w%crop_size!=0 and h%crop_size!=0:
        crop_imgs.append(img[h-512:h,w-512:w,:])
        crop_jpe_dcts.append(jpg_dct[h-512:h,w-512:w])
        if use_mask:
            if mask[h-512:h,w-512:w].max()!=0:
                crop_masks.append(1)
            else:
                crop_masks.append(0)

    if use_mask:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, crop_masks
    else:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, None


def combine_img(imgs, h_grids, w_grids, img_h, img_w, crop_size=512):
    i = 0
    re_img = np.zeros((img_h, img_w))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, x1:x2] = imgs[i]
            i += 1

    if w_grids*crop_size<img_w:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2,img_w-512:img_w]=imgs[i]
            i+=1

    if h_grids*crop_size<img_h:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            re_img[img_h-512:img_h,x1:x2]=imgs[i]
            i+=1

    if w_grids*crop_size<img_w and h_grids*crop_size<img_h:
        re_img[img_h-512:img_h,img_w-512:img_w] = imgs[i]

    return re_img

def make_square(path):
    img = cv2.imread(path)
    h,w,c = img.shape
    if h > w:
        dw = h - w
        dh = 0
    elif h < w:
        dw = 0
        dh = w - h
    else:
        dh = 0
        dw = 0
    if dh!=0 or dw!=0:
        img = np.pad(img,((0,dh),(0,dw),(0,0)),'constant',constant_values=255)
        img = Image.fromarray(img).convert("L")
        img.save(path,"JPEG",qtables={0:new_qtb})
    return
data_path = 'test_images'
result_path = 'results'
def process_image_with_model(img_path, model):

    totsr = ToTensorV2()
    toctsr =torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
        ])
    model.eval()
    crop_masks_alls = []
    pred_lists_alls = []
    if not str(path).endswith(("jpg", 'jpeg', 'JPG', 'JPEG')):
        img = cv2.imread(img_path)
        img = Image.fromarray(img).convert("L")
        img.save(img_path,"JPEG",qtables={0:new_qtb})
    make_square(img_path)
    imgs_ori = cv2.imread(img_path)
    h,w,c = imgs_ori.shape
    jpg_dct = jpegio.read(img_path)
    gt_mask = cv2.imread('test_masks/'+path[:-4]+'.png',0)
    dct_ori = jpg_dct.coef_arrays[0].copy()
    use_qtb2 = jpg_dct.quant_tables[0].copy()
    if min(h,w)<512:
        H,W = gt_mask.shape[:2]
        if H < 512:
            dh = (512-H)
        else:
            dh = 0
        if W < 512:
            dw = (512-W)
        else:
            dw = 0
        imgs_ori = np.pad(imgs_ori,((0,dh),(0,dw),(0,0)),'constant',constant_values=255)
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            imgs_ori = Image.fromarray(imgs_ori).convert("L")
            imgs_ori.save(tmp,"JPEG",qtables={0:new_qtb})
            jpg = jpegio.read(tmp.name)
            dct_ori = jpg.coef_arrays[0].copy()
            imgs_ori = np.array(imgs_ori.convert('RGB'))
            use_qtb2 = jpg.quant_tables[0].copy()
        h,w,c = imgs_ori.shape   

    if h%8 == 0 and w%8 == 0:
        imgs_d = imgs_ori
        dct_d = dct_ori
    else:
        imgs_d = imgs_ori[0:(h//8)*8,0:(w//8)*8,:].copy()
        dct_d = dct_ori[0:(h//8)*8,0:(w//8)*8].copy()

    qs = torch.LongTensor(use_qtb2)
    img_h, img_w, _ = imgs_d.shape
    crop_imgs, crop_jpe_dcts, h_grids, w_grids, _= crop_img(imgs_d, dct_d, crop_size=512, mask=gt_mask)
    img_list = []
    for idx, crop in enumerate(crop_imgs):
        crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        data = toctsr(crop)
        dct = torch.LongTensor(crop_jpe_dcts[idx])

        data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
        dct = torch.abs(dct).clamp(0,20)
        B,C,H,W = data.shape
        qs = qs.reshape(B,1,8,8)
        with torch.no_grad():
            if data.size()[-2:]==torch.Size((512,512))  and dct.size()[-2:]==torch.Size((512,512)) and qs.size()[-2:]==torch.Size((8,8)):
                pred = model(data,dct,qs)
                pred = torch.nn.functional.softmax(pred,1)[:,1].cpu()
                img_list.append(((pred.cpu().numpy())*255).astype(np.uint8))
    ci = combine_img(img_list, h_grids, w_grids, img_h, img_w, crop_size=512)
    padding = (0, 0, w-img_w, h-img_h)
    ci = cv2.copyMakeBorder(ci, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite("mask.png", ci)

    # read the mask and put it on the original image
    mask = cv2.imread("mask.png", 0)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgs_ori, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite("temp_bbox.jpg", imgs_ori)
    final = cv2.imread("temp_bbox.jpg")
    return final

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
    st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)
    add_forgery_icons()
    st.write("Welcome to our state-of-the-art Forgery Detection system. Upload an image to check for potential manipulations.")
    add_info_boxes()

add_image_comparison()

model = load_model()

if uploaded_file is not None:
    # write the uploaded image

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(data_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_column_width=True)

    # save the uploaded image
    # image = cv2.imread(os.path.join(data_path, uploaded_file.name))
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # save the uploaded image
    path = os.path.join(data_path, uploaded_file.name)
    # image = Image.fromarray(image).convert("L")
    # image.save(path,"JPEG",qtables={0:new_qtb})

    with st.spinner("Analyzing for potential forgery..."):
        processed_image = process_image_with_model(path, model)

    # save the analyzed image
    processed_image_path = os.path.join(result_path, uploaded_file.name)
    cv2.imwrite(processed_image_path, processed_image)

    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    st.image(processed_image_rgb, caption="Analyzed Image", use_column_width=True)
    st.download_button("Download Analyzed Image", data=Image.fromarray(processed_image_rgb).tobytes(), file_name="analyzed_image.png", mime="image/png")

st.markdown('</div>', unsafe_allow_html=True)