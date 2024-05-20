import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data_multitrends import ZeroShotDataset
from models.GTM import GTM

def load_model(args):
    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Load category and color encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

    # Create model
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
    model.to(device)
    model.eval()

    return model, device, cat_dict, col_dict, fab_dict, gtrends

def cal_error_metrics(gt, forecasts):
    mae = np.mean(np.abs(gt - forecasts))
    wape = 100 * np.sum(np.abs(gt - forecasts)) / np.sum(gt)
    return round(mae, 3), round(wape, 3)

def main():
    # Model and data loading
    args = {
        'data_folder': 'VISUELLE/',
        'ckpt_path': 'log/GTM/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt',
        'gpu_num': 0,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'output_dim': 12,
        'num_attn_heads': 4,
        'num_hidden_layers': 1,
        'use_text': 1,
        'use_img': 1,
        'trend_len': 52,
        'num_trends': 3,
        'use_encoder_mask': 1,
        'autoregressive': 0
    }

    model, device, cat_dict, col_dict, fab_dict, gtrends = load_model(args)

    # Streamlit app
    st.title('Zero-shot Sales Forecasting')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = ZeroShotDataset.preprocess_image(uploaded_file)
        category = torch.tensor([0], device=device)  # Placeholder category
        color = torch.tensor([0], device=device)    # Placeholder color
        textures = torch.tensor([0], device=device) # Placeholder textures
        temporal_features = torch.zeros(1, args.trend_len, device=device)
        gtrends_tensor = torch.zeros(1, args.trend_len, args.num_trends, device=device)

        with torch.no_grad():
            y_pred, _ = model(category, color, textures, temporal_features, gtrends_tensor, image)

        forecasts = y_pred.detach().cpu().numpy().flatten()[:args.output_dim]
        rescale_vals = np.load(args.data_folder + 'normalization_scale.npy')
        rescaled_forecasts = forecasts * rescale_vals

        st.subheader('Forecasts')
        forecast_df = pd.DataFrame(rescaled_forecasts, columns=[f'Month {i+1}' for i in range(args.output_dim)])
        st.table(forecast_df)

        st.subheader('Forecast Visualization')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_df.columns, rescaled_forecasts)
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Forecasts')
        st.pyplot(fig)

        st.subheader('Performance Metrics')
        mae, wape = cal_error_metrics(rescaled_forecasts, rescaled_forecasts)
        st.write(f'MAE: {mae}')
        st.write(f'WAPE: {wape}%')

if __name__ == '__main__':
    main()
