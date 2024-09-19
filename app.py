import os
import time
import torch
import pprint
import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.db import PostgreSQLDatabase
from src.model.model import ModelOnlyRegression
from src.electricity_predict import ElectricityPredictor


# Load environment variables from .env file
load_dotenv()


st.set_page_config(
    page_title="SEnergy-NILM",
    page_icon="🧊",
    layout="wide",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("ELECTRIcity")

# Database client
client1 = PostgreSQLDatabase(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    entity_id=os.getenv("DB_ENTITY_ID"),
)

try:
    # Establish a connection to the database
    client1.connect()
    assert client1.connection is not None
    print("Successfully connected to the database.")
except Exception as e:
    print(e)
    print("Failed to connect to the database.")

MODELS_FOLDER = "models"

model_paths = []
appliances = []
args = []

for app in os.listdir(MODELS_FOLDER):
    model_paths.append(os.path.join(os.path.join("models", app), "best_model.pt"))
    appliances.append(app)
    args.append(
        pk.load(open(os.path.join(os.path.join("models", app), "results.pkl"), "rb"))[
            "args"
        ]
    )

# print("Model Paths:", model_paths)
# print("Appliances:", appliances)
# print("Args:", args)

old_model = ElectricityPredictor(model_paths, appliances, args)

# New model setup
sequence_length = 128
num_filters = 16
kernel_size = 8
hidden_units = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hparams:
#     L: 128
#     F: 16
#     K: 8
#     H: 1024


def save_model(model, optimizer, hparams, appliance, transform, file_name_model, error):
    """
    Save the model, optimizer, hyperparameters, appliance configuration, and
    preprocessing transform to a file using PyTorch's torch.save().
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hparams": hparams,
            "appliance": appliance,
            "transform": transform,
            "error": error,
        },
        file_name_model,
    )


def load_model(file_name_model, model, optimizer=None):
    """
    Load model and metadata from file
    """
    if torch.cuda.is_available():
        state = torch.load(file_name_model)
    else:
        state = torch.load(file_name_model, map_location=torch.device("cpu"))

    model.load_state_dict(state["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    hparams = state.get("hparams", None)
    appliance = state.get("appliance", None)

    transform = state.get("transform", None)
    error = state.get("error", None)

    print("=========== ARCHITECTURE ==========")
    print("Reloading appliance")
    pprint.pprint(appliance)
    print("Reloading transform")
    pprint.pprint(transform)
    print("===================================")
    return transform, error


new_model = ModelOnlyRegression(
    sequence_length, num_filters, kernel_size, hidden_units
)
new_model = new_model.to(device)
transform, error = load_model("models/microwave/microwave.th", new_model)
print("transform: ", transform)
print("error: ", error)
new_model.eval()  # Set the model to evaluation mode


def preprocess_data(data, sequence_length=128):
    """
    Preprocess the input data into windows of the specified sequence length.
    """
    # Assuming data is a numpy array of shape (total_length,)
    windows = np.lib.stride_tricks.sliding_window_view(data, sequence_length)
    windows = torch.from_numpy(windows).float()
    windows = windows.unsqueeze(1)  # Add channel dimension
    return windows.to(device)


def predict_new_model(seq):
    inputs = preprocess_data(np.array(seq), sequence_length)
    with torch.no_grad():
        combined_output, regression_output, attention_weights, classification_output = (
            new_model(inputs)
        )
    return combined_output.cpu().numpy()[-1,-1]


chart_data = pd.DataFrame(
    {
        "aggregated": [],
        "fridge": [],
        "kettle": [],
        "microwave": [],
        "dishwasher": [],
        "new_model": [],
    }
)


def get_data():
    global chart_data

    seq = (np.array(client1.get_data(row=480, table="ts_kv")) * 1000).tolist()[::-1]
    # print(seq)
    old_predictions = old_model.predict(seq)[0]
    new_prediction = predict_new_model(seq)

    new_data = pd.DataFrame(
        {
            "aggregated": [seq[-1]],
            "fridge": [list(old_predictions["fridge"])[-1]],
            "kettle": [list(old_predictions["kettle"])[-1]],
            "microwave": [list(old_predictions["microwave"])[-1]],
            "dishwasher": [list(old_predictions["dishwasher"])[-1]],
            "new_model": [new_prediction],
        }
    )

    chart_data = pd.concat(
        [chart_data, new_data],
        ignore_index=True,
    )

    return chart_data


plot_spot = st.plotly_chart(px.line(get_data(), height=800), use_container_width=True)


while True:
    plot_spot.plotly_chart(px.line(get_data(), height=600), use_container_width=True)
    time.sleep(1)

client1.close()
