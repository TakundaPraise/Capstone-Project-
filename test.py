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
import base64

# Set up Streamlit app
st.set_page_config(page_title="Capstone Project ")

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

model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Streamlit UI
st.title("PREDICTIVE ORDERING FOR NEW FASHION PRODUCTS ")

welcome_image = "fashion3.jpeg"
st.image(welcome_image, caption="", width=400)

# Add an overview of how the system works
st.sidebar.subheader("System Overview")
st.sidebar.write("This system utilizes a zero-shot sales predictions approach. It takes into account various inputs such as image features, category, color, fabric, temporal features, and Google Trends data to generate sales predictions for new fashion products to help fashion retailers when doing ordering.")

# File upload
st.write('please upload an i age of file type JPG, JPEG, or PNG')
uploaded_file = st.sidebar.file_uploader("Choose an image",  accept_multiple_files=False)

if uploaded_file is not None:
    # Check the file size
    if uploaded_file.size > 209715200:  # 200 MB in bytes
        st.error("File size exceeds the limit of 200 MB.")
    elif uploaded_file.type not in ["image/jpeg", "image/jpg", "image/png"]:
        st.error("Invalid file type. Please upload an image file (JPG, JPEG, or PNG).")
    else:
        # Preprocess the uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        img_transforms = Compose([Resize((40, 40)), ToTensor(), Normalize(mean=[0.012, 0.010, 0.008], std=[0.029, 0.024, 0.025])])
        image_feature = img_transforms(img).unsqueeze(0).to(device)

        # Generate forecasts
        with torch.no_grad():
            category = torch.tensor([0], device=device)
            color = torch.tensor([0], device=device)
            fabric = torch.tensor([0], device=device)
            temporal_features = torch.zeros(1, 4).to(device)
            gtrends = torch.zeros(1, 3, 52).to(device)

            y_pred, _ = model(category, color, fabric, temporal_features, gtrends, image_feature)

            # Rescale the forecasts
            rescale_vals = np.load('VISUELLE/normalization_scale.npy')
            rescaled_forecasts = y_pred.detach().cpu().numpy().flatten()[:12] * rescale_vals

            # Round the forecasts to whole numbers
            rounded_forecasts = np.round(rescaled_forecasts).astype(int)
            week_labels = [f'Week {i+1}' for i in range(12)]

        # Display the forecasts
        st.subheader("NEW PRODUCTS SALES PREDICTIONS LINE CHART")
        chart_expander = st.expander("Click to expand", expanded=False)
        with chart_expander:
            st.write('The line chart represents the sales predictions for new fashion products. The y-axis represents the sales predictions. The chart displays a line that indicates the predicted sales over time. The values on the y-axis represent the estimated number of sales for each corresponding time point on the x-axis. The line connects these data points, showing the trend or pattern in the sales predictions. The predictions are then rescaled using normalization values and presented in both the line chart and the table.')
            week_labels = [f'Week {i+1}' for i in range(12)]

            # Create the line chart with updated labels
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rounded_forecasts)
            ax.set_xticks(range(len(week_labels)))
            ax.set_xticklabels(week_labels, rotation=90)
            ax.set_xlabel('Week')
            ax.set_ylabel('Sales Predictions')
            st.pyplot(fig)

        st.subheader("NEW PRODUCTS SALES PREDICTIONS TABLE")
        forecast_df = pd.DataFrame(rounded_forecasts, columns=['Sales'])
        forecast_df.index = week_labels
        chart_expander = st.expander("Click to expand", expanded=False)
        with chart_expander:
            st.write('The table represents the sales predictions for the new fashion products. The values in the column 0 represent the months, and the values in the other column represent the sales predictions of the products. The table displays the values of the sales corresponding to the months.')
            st.table(forecast_df)
