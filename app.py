import os
import time
import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.db import PostgreSQLDatabase
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

print("Model Paths:", model_paths)
print("Appliances:", appliances)
print("Args:", args)

model = ElectricityPredictor(model_paths, appliances, args)


chart_data = pd.DataFrame(
    {
        "aggregated": [],
        "fridge": [],
        "kettle": [],
        "microwave": [],
        "dishwasher": [],
    }
)


def get_data():
    global chart_data

    seq = (np.array(client1.get_data(row=480, table="ts_kv")) * 1000).tolist()[::-1]
    # print(seq)

    new_data = pd.DataFrame(
        {
            "aggregated": [seq[-1]],
            "fridge": [list(model.predict(seq)[0]["fridge"])[-1]],
            "kettle": [list(model.predict(seq)[0]["kettle"])[-1]],
            "microwave": [list(model.predict(seq)[0]["microwave"])[-1]],
            "dishwasher": [list(model.predict(seq)[0]["dishwasher"])[-1]],
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
