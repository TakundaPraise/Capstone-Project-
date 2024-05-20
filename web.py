import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from models.GTM import GTM
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.data_multitrends import ZeroShotDataset

def load_model():
    # Initialize the device variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    cat_dict = torch.load(Path('VISUELLE/category_labels.pt'))
    col_dict = torch.load(Path('VISUELLE/color_labels.pt'))
    fab_dict = torch.load(Path('VISUELLE/fabric_labels.pt'))

    model = GTM(
        embedding_dim=32,
        hidden_dim=64,
        output_dim=12,
        num_heads=4,
        num_layers=1,
        use_text=1,
        use_img=1,
        trend_len=52,
        num_trends=3,
        use_encoder_mask=1,
        autoregressive=0,
        gpu_num=0,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
    )

    # Load the model weights
    model_path = 'log/GTM/GTM_experiment2---epoch=29---16-05-2024-08-49-43.ckpt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)
    return model, device

    

    # Remove the extra keys
    for key in list(state_dict.keys()):
        if not key.startswith("module.") and "dummy_encoder" in key:
            del state_dict[key]
        if key in ["epoch", "global_step", "pytorch-lightning_version", "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers", "hparams_name", "hyper_parameters"]:
            del state_dict[key]

    # Load the state dict
    model.load_state_dict(state_dict)
    print("Model loaded successfully")
    model.eval()

    # Get the device that the model is loaded on
    device = next(model.parameters()).device

    return model, device
    

def preprocess_image(image_path, device):
    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Resize and normalize the image
    img_transforms = Compose([
        Resize((40, 40)),
        ToTensor(),
        Normalize(mean=[0.012, 0.010, 0.008], std=[0.029, 0.024, 0.025])
    ])
    image = img_transforms(image)

    # Add a batch dimension
    image = image.unsqueeze(0).to(device)

    return image

def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = np.mean(np.abs(gt - forecasts))
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return mae, wape

def run_forecast(model, image_path, device, cat_dict, col_dict, fab_dict, gtrends, rescale_vals):
    # Preprocess the input image
    image_path = uploaded_file
    image = preprocess_image(image_path, device)
    batch = image.unsqueeze(0)
    category = torch.tensor([0], device=device)  # Placeholder category
    color = torch.tensor([0], device=device)    # Placeholder color
    textures = torch.tensor([0], device=device) # Placeholder textures
    temporal_features = torch.zeros(1, 52, device=device)
    gtrends_tensor = torch.zeros(1, 52, 3, device=device)

    # Generate forecasts
    with torch.no_grad():
        #y_pred, _ = model(category, color, textures, temporal_features, gtrends_tensor, image)
        y_pred, _ = model(category, color, textures, temporal_features, gtrends_tensor, batch)
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
