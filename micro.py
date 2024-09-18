import os
import torch
import pprint
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.db import PostgreSQLDatabase
from src.model.model import ModelOnlyRegression

load_dotenv()
sequence_length =128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Database client
client1 = PostgreSQLDatabase(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    entity_id=os.getenv("DB_ENTITY_ID"),
)

# Get real-world data
client1.connect()
real_world_data = (np.array(client1.get_data(row=480, table="ts_kv")) * 1000).tolist()[
    ::-1
]

client1.close()

# Convert to numpy array if it's not already
real_world_data = np.array(real_world_data)
# real_world_data = np.array(real_world_data, dtype=np.float64) * 1000

print("=============> Real-world data:", real_world_data[sequence_length:])


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


def preprocess_data(data, sequence_length=128):
    """
    Preprocess the input data into windows of the specified sequence length.
    """
    # Assuming data is a numpy array of shape (total_length,)
    windows = np.lib.stride_tricks.sliding_window_view(data, sequence_length)
    windows = torch.from_numpy(windows).float()
    windows = windows.unsqueeze(1)  # Add channel dimension
    return windows.to(device)


# hparams:
#     L: 128
#     F: 16
#     K: 8
#     H: 1024

model = ModelOnlyRegression(128, 16, 8, 1024)
model = model.to(device)
transform, error = load_model("models/microwave/microwave.th", model)
print("transform: ", transform)
print("error: ", error)
model.eval()  # Set the model to evaluation mode


# Preprocess the data
inputs = preprocess_data(real_world_data, sequence_length=128)
print("inputs: ", inputs)

# Run inference
with torch.no_grad():
    combined_output, regression_output, attention_weights, classification_output = model(inputs)
    print("combined_output: ", combined_output)

# Get predictions
predictions = combined_output.cpu().numpy()
print("=============================== Predictions ===============================")
print("predictions: ", predictions)
print("==========================================================================")




plt.figure(figsize=(12, 6))
plt.plot(real_world_data[sequence_length:], label='Original')
plt.plot(predictions[:, -1], label='Predicted')
plt.legend()
plt.title('Original vs Predicted Power Consumption')
plt.show()