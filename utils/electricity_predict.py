import torch
import pickle
import copy
import numpy as np
import pandas as pd
import os

# from .Electricity_model import *


class ELECTRIcity:

    def __init__(self, model_paths, appliances, args):
        self.appliances = appliances
        self.args = args
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.models = {}
        
        for p in range(len(self.appliances)):
            electricity = torch.load(model_paths[p], map_location=self.device)
            electricity.eval()
            self.models[self.appliances[p]] = electricity.double()

        self.window_size = args[0].window_size

        self.device = torch.device("cpu")

        self.cutoff = args[0].cutoff
        self.threshold = args[0].threshold

    def get_model_outputs(self, model, x, cutoff, threshold, mask=None):
        logits, gen_out = model(x, mask)
        logits_y = self.cutoff_energy(logits * cutoff, cutoff)
        logits_status = self.compute_status(logits_y, threshold)
        return logits, gen_out, logits_y, logits_status

    def cutoff_energy(self, data, cutoff):
        data[data < 5] = 0
        data = torch.min(data, cutoff.double())
        return data

    def denoise_data(self, data):
        smooth_data = pd.Series(data).rolling(window=5).median().fillna(method="bfill")
        return smooth_data.values

    def compute_status(self, data, threshold):
        status = (data >= threshold) * 1
        return status

    def predict(self, input_seq):
        seq = np.array(input_seq, dtype=np.float64) * 1000

        appliances_power = {}
        appliances_stats = {}

        for i in range(len(self.appliances)):
            agg = torch.tensor(seq, dtype=torch.float64, device=self.device)

            mean = torch.tensor(self.args[i]._mean).to(self.device)
            std = torch.tensor(self.args[i]._std).to(self.device)

            agg = (agg - mean) / std
            agg = agg.unsqueeze(0).unsqueeze(0)

            cutoff = torch.tensor(self.cutoff[self.appliances[i]]).to(self.device)
            threshold = torch.tensor(self.threshold[self.appliances[i]]).to(self.device)

            _, _, y, s = self.get_model_outputs(self.models[i], agg, cutoff, threshold)
            y = y * s

            appliances_power[self.appliances[i]] = y.detach().cpu().numpy().squeeze()
            appliances_stats[self.appliances[i]] = s.detach().cpu().numpy().squeeze()

        return appliances_power, appliances_stats
