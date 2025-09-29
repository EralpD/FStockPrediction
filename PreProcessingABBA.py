import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from slearn.symbols import *
from fABBA import compress, digitize, inverse_compress, inverse_digitize
from dataToday import getData

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # We will use LabelEncoder to classify ABBA strings

def preprocess(ticker:str, day_intv:str, block_size, split, pred_length, tol, scl=1, alp=.1):

    data, today = getData(ticker, day_intv)
    data = data.reset_index()
    data.columns = [col[0] for col in data.columns]

    data_abba = data['Close'].values.reshape(-1, 1)
    pieces = compress(data_abba, tol=tol)
    string, parameters = digitize(pieces, scl=scl, alpha=alp)
    # print(len(string)) # 1096+ ✓

    # Creating labels respects alphabets
    labelEncoder = LabelEncoder()
    labelEncoder.fit(parameters.alphabets)
    labels = labelEncoder.transform(string)

    # print(len(labelEncoder.classes_)) # 136+ ✓
    vocab_size = len(labelEncoder.classes_)
    # print(labels.shape) # 1096+ ✓

    block_size = 32
    pred_length = 20
    split = 0.8

    x, y = [], []

    # build all possible (X, Y) pairs
    for i in range(block_size + pred_length, len(labels)):
        x.append(labels[i - block_size - pred_length : i - pred_length])
        y.append(labels[i - pred_length : i])

    x = np.array(x)  # shape: (#samples, block_size, features)
    y = np.array(y)  # shape: (#samples, pred_length, features)

    # split index
    split_idx = int(len(x) * split)

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor  = torch.tensor(x_test, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    # datasets + loaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(x_train_tensor.shape)
    print(x_test_tensor.shape)
    print(y_train_tensor.shape)
    print(y_test_tensor.shape)
    
    return data, (block_size, split, pred_length), (train_loader, test_loader), (string, parameters, pieces), vocab_size