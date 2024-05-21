import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from pathlib import Path
from models.gtm import GTM
from models.FCN import FCN
import matplotlib.pyplot as plt


# Set up Streamlit app
st.set_page_config(page_title="Zero-shot Sales Forecasting")

# Load category and color encodings
cat_dict = torch.load(Path('VISUELLE/category_labels.pt'))
col_dict = torch.load(Path('VISUELLE/color_labels.pt'))
fab_dict = torch.load(Path('VISUELLE/fabric_labels.pt'))

# Load Google trends
gtrends = pd.read_csv(Path('VISUELLE/gtrends.csv'), index_col=[0], parse_dates=True)

# Load pre-trained model
model_type = 'GTM'
model_savename = f'experiment2_12'
ckpt_path = 'log/GTM/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt'

if model_type == 'FCN':
    model = FCN(
        embedding_dim=32,
        hidden_dim=64,
        output_dim=12,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_trends=1,
        use_text=1,
        use_img=1,
        trend_len=52,
        num_trends=3,
        use_encoder_mask=1,
        gpu_num=0
    )
else:
    model = GTM(
        embedding_dim=32,
        hidden_dim=64,
        output_dim=12,
        num_heads=4,
        num_layers=1,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=1,
        use_img=1,
        trend_len=52,
        num_trends=3,
        use_encoder_mask=1,
        autoregressive=0,
        gpu_num=0
    )
#orch.load with map_location=torch.device('cpu') 

model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Streamlit UI
# Add a background image to the home page
background_image = Image.open('fashion3.jpeg')
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/jpeg;base64,{base64.b64encode(background_image.read()).decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("PREDICTIVE ORDERING FOR NEW FASHION PRODUCTS ")
st.markdown('<p style="color:blue;font-size:30px;">Zero-shot Sales Forecasting</p>', unsafe_allow_html=True)


# Add an overview of how the system works
st.subheader("System Overview")
st.write("This system utilizes a zero-shot sales predictions approach. It takes into account various inputs such as image features, category, color, fabric, temporal features, and Google Trends data to generate sales predictions for new fashion products to help fashion retailers when doing ordering .")

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    img_transforms = Compose([Resize((40, 40)), ToTensor(), Normalize(mean=[0.012, 0.010, 0.008], std=[0.029, 0.024, 0.025])])
    image_feature = img_transforms(img).unsqueeze(0).to(device)

    # Generate forecasts
    with torch.no_grad():
        #category = torch.tensor([0], device=device)  # Placeholder category
        #category = torch.LongTensor([cat_dict['shirt']]).to(device)
        #color = torch.LongTensor([col_dict['blue']]).to(device
        #fabric = torch.LongTensor([fab_dict['cotton']]).to(device)
        category = torch.tensor([0], device=device)  # Placeholder category
        color = torch.tensor([0], device=device)    # Placeholder color
        fabric = torch.tensor([0], device=device) # Placeholder textures
        temporal_features = torch.zeros(1, 4).to(device)
        gtrends = torch.zeros(1, 3, 52).to(device)

        y_pred, _ = model(category, color, fabric, temporal_features, gtrends, image_feature)

        # Rescale the forecasts
        rescale_vals = np.load('VISUELLE/normalization_scale.npy')
        rescaled_forecasts = y_pred.detach().cpu().numpy().flatten()[:12] * rescale_vals

    # Display the forecasts
    st.subheader("NEW PRODUCTS SALES PREDICTIONS LINE CHART ")
    st.line_chart(rescaled_forecasts)

    st.subheader("NEW PRODUCTS SALES PREDICTIONS TABLE")
    forecast_df = pd.DataFrame(rescaled_forecasts, columns=['Sales'])
    st.table(forecast_df)

    
