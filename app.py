import streamlit as st
import os
import torch
import pandas as pd
import argparse

import numpy as np
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from sklearn.metrics import mean_absolute_error

def main():
    st.title("Zero-Shot Sales Forecasting")

    # Load model and configuration
    parser = argparse.ArgumentParser(description='Zero-Shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='VISUELLE/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='log/GTM\GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt')
    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    args = parser.parse_args()

    model = load_model(args)

    # Streamlit app
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_path = os.path.join('uploads', uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        forecasts = forecast(model, img_path, args)
        rescale_vals = np.load(os.path.join(args.data_folder, 'normalization_scale.npy'))
        rescaled_forecasts = forecasts * rescale_vals

        st.header("Forecasts")
        forecasts_df = pd.DataFrame(rescaled_forecasts, columns=[f"Week {i+1}" for i in range(args.output_dim)])
        st.table(forecasts_df)

        st.header("Error Metrics")
        mae, wape = cal_error_metrics(np.ones(args.output_dim) * rescaled_forecasts.mean(), rescaled_forecasts)
        st.write(f"MAE: {mae}")
        st.write(f"WAPE: {wape}")

        st.header("Forecast Plot")
        st.line_chart(forecasts_df)

if __name__ == '__main__':
    main()
