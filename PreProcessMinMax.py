import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from dataToday import getData

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def preProcess(ticker:str, day_intv:str, block_size, split, pred_length):
    data, today = getData(ticker, day_intv)
    data = data.reset_index()
    data.columns = [col[0] for col in data.columns]

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = data['Close'].iloc[:int(split*len(data))].values.reshape(-1, 1)
    test_close = data['Close'].iloc[int(split*len(data)):].values.reshape(-1, 1)
    scaler.fit(train_close)

    data['Close_scaled'] = scaler.transform(data['Close'].values.reshape(-1, 1))
    data_scaled = data['Close_scaled'].values.reshape(-1, 1)

    e = 0
    for i in range(block_size+pred_length, len(data)):
        if len(y_train) < int(split * len(data))-block_size-pred_length:
            x_train.append(data_scaled[i-block_size-pred_length:i-pred_length])
            y_train.append(data_scaled[i-pred_length:i])
        else:
            if e < block_size+pred_length:
                x_train.append(data_scaled[i-block_size-pred_length:i-pred_length])
                y_train.append(data_scaled[i-pred_length:i])
                e += 1
            x_test.append(data_scaled[i-block_size-pred_length:i-pred_length])
            y_test.append(data_scaled[i-pred_length:i])


    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)

    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    print(f"Shape of x_train: {x_train_tensor.shape}")
    print(f"Shape of x_test: {x_test_tensor.shape}")
    print(f"Shape of y_train: {y_train_tensor.shape}")
    print(f"Shape of y_test: {y_test_tensor.shape}")

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return data, (block_size, split, pred_length), (train_loader, test_loader)