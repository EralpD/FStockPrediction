import torch
import matplotlib.pyplot as plt
import numpy as np
from PreProcessMinMax import preProcess
from LSTMAttnSequential import LSTMAttentionModel
import pandas as pd
import pandas_market_calendars as mcal
import os
import matplotlib.dates as pldates


prompt = input("Please enter ticker and format for prediction. (AAPL, 1; NVDA, 20;\'0 (e.g. single prediction to AAPL, 4 week prediction for NVDA)):\n")

prompt = prompt.split(";")[:-1]
try:
    for pmp in prompt:
        piece = pmp.split(",")
        ticker = piece[0].strip()
        pred_len = int(piece[1].strip())

        model = LSTMAttentionModel(1, 209, pred_len, num_head=1)
        if pred_len == 1:
            model.load_state_dict(torch.load("Resources/modelS.pt"))
        elif pred_len == 20:
            model.load_state_dict(torch.load("Resources/model.pt"))
        else:
            raise TypeError()
        
        data, scaler, (block_size, _, pred_len), (_, _) = preProcess(ticker, "1d", block_size=32, split=.8, pred_length=pred_len)

        last_window = data['Close_scaled'].values[-block_size:]
        last_window = last_window = torch.tensor(last_window, dtype=torch.float32).view(1, block_size, 1)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_window)

        pred = scaler.inverse_transform(pred_scaled.numpy())

        if pred_len == 20:
            offset = data['Close'].iloc[-1] - pred[0, 0]
            pred = pred + offset
        elif pred_len == 1:
            pass
        
        nyse = mcal.get_calendar('NYSE')
        last_date = data['Date'].iloc[-1]

        future_days = nyse.valid_days(start_date=last_date, end_date=last_date + pd.Timedelta(days=pred_len*3))[:pred_len]
        pred_array = np.array(pred).flatten()

        plt.plot(data['Date'], data['Close'], color="red", label="Market")
        if pred_len == 1:
            plt.scatter(future_days[0], pred_array[0], color='yellow', s=25, zorder=3)  # highlight point
            plt.text(
            future_days[0], pred_array[0], f"${pred_array[0]:.2f}$", 
            color='yellow', fontsize=10, fontweight='bold',
            ha='left', va='bottom'
            )
        else:
            plt.plot(future_days, pred_array, color="blue", label="Prediction", alpha=.5)
        plt.xlabel("Date", color="White")
        plt.ylabel("Price (USD)", color="White")
        title = ticker + " Daily Close Prediction (2020–2025+)" if pred_len == 1 else ticker + " Monthly Close Prediction (2020–2025+)"
        plt.title(title, color="White")
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.set_major_locator(pldates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(pldates.DateFormatter('%Y-%m'))
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)  # Making spines thicker
            spine.set_edgecolor('white')

        fig.set_size_inches(15, 6)
        fig.patch.set_facecolor('black')
        plt.legend(loc="upper left")
        path = "Resources/Predictions/" + ticker + str(pred_len) + ".png"
        if not os.path.exists(path):
            fig.savefig(path, dpi=300)
        plt.show()
            
except:
    print("Please check your input format. (The sucessful ones already saved on Resources/Predictions file)")
    