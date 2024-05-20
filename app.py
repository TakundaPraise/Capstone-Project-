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

def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)
    return round(mae, 3), round(wape, 3)

def load_model(args):
    # Load category and color encodings
    cat_dict = torch.load(os.path.join(args.data_folder, 'category_labels.pt'))
    col_dict = torch.load(os.path.join(args.data_folder, 'color_labels.pt'))
    fab_dict = torch.load(os.path.join(args.data_folder, 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(os.path.join(args.data_folder, 'gtrends.csv'), index_col=[0], parse_dates=True)

    # Create model
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    return model

def forecast(model, img_path, args):
    # Load and preprocess the image
    img_transforms = Compose([Resize((10, 10)), ToTensor(), Normalize(mean=[0.012, 0.010, 0.008], std=[0.029, 0.024, 0.025])])
    img = Image.open(img_path).convert('RGB')
    img = img_transforms(img).unsqueeze(0)

    # Get the product information
    category = torch.tensor([args.cat_dict['category']])
    color = torch.tensor([args.col_dict['color']])
    fabric = torch.tensor([args.fab_dict['fabric']])
    temporal_features = torch.zeros(1, 4)
    gtrends = torch.zeros(1, args.num_trends, args.trend_len)

    # Forward pass
    with torch.no_grad():
        y_pred, _ = model(category, color, fabric, temporal_features, gtrends, img)

    return y_pred.detach().cpu().numpy().flatten()[:args.output_dim]

def main():
    st.title("Zero-Shot Sales Forecasting")

    # Load model and configuration
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='VISUELLE/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu_num', type=int, default=0)

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
