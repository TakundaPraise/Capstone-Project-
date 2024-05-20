import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset

def load_model():
    # Load the model
    model = GTM(
        embedding_dim=32,
        hidden_dim=64,
        output_dim=12,
        num_heads=4,
        num_layers=1,
        use_text=1,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_img=1,
        trend_len=52,
        num_trends=3,
        use_encoder_mask=1,
        autoregressive=0,
        gpu_num=0
    )
    model_path = 'log/GTM/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt'
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    print("Model loaded successfully")
    model.eval()

    # Get the device that the model is loaded on
    device = next(model.parameters()).device

    return model, device
def preprocess_image(image, device):
    # Preprocess the input image
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.
    image = np.transpose(image, [2, 0, 1])
    image = torch.tensor(image.copy(), device=device)
    image = (image - 0.5) / 0.5

    return image.unsqueeze(0)

def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = np.mean(np.abs(gt - forecasts))
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return mae, wape

def run_forecast(model, image, device, cat_dict, col_dict, fab_dict, gtrends, rescale_vals):
    # Preprocess the input image
    image = preprocess_image(image, device)
    category = torch.tensor([0], device=device)  # Placeholder category
    color = torch.tensor([0], device=device)    # Placeholder color
    textures = torch.tensor([0], device=device) # Placeholder textures
    temporal_features = torch.zeros(1, 52, device=device)
    gtrends_tensor = torch.zeros(1, 52, 3, device=device)

    # Generate forecasts
    with torch.no_grad():
        y_pred, _ = model(category, color, textures, temporal_features, gtrends_tensor, image)

    forecasts = y_pred.detach().cpu().numpy().flatten()[:12]
    rescaled_forecasts = forecasts * rescale_vals

    return rescaled_forecasts

if __name__ == '__main__':
    # Model and data loading
    model, device = load_model()

    # Load category and color encodings
    cat_dict = torch.load(Path('VISUELLE/category_labels.pt'))
    col_dict = torch.load(Path('VISUELLE/color_labels.pt'))
    fab_dict = torch.load(Path('VISUELLE/fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path('VISUELLE/gtrends.csv'), index_col=[0], parse_dates=True)

    # Load normalization scale
    rescale_vals = np.load(Path('VISUELLE/normalization_scale.npy'))

    # Streamlit app
    import streamlit as st

    st.title('Zero-shot Sales Forecasting')

    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Generate forecasts
        rescaled_forecasts = run_forecast(model, image, device, cat_dict, col_dict, fab_dict, gtrends, rescale_vals)

        # Display the rescaled forecasts
        st.subheader('Forecasts')
        forecast_df = pd.DataFrame(rescaled_forecasts.reshape(1, 12), columns=[f'Month {i+1}' for i in range(12)])
        st.table(forecast_df)

        st.subheader('Forecast Visualization')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_df.columns, rescaled_forecasts)
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales')
        ax.set_title('Sales Forecasts')
        st.pyplot(fig)
