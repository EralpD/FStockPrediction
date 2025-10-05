import time
import torch
from PreProcessMinMax import preProcess
import torch.nn as nn
import pytorch_forecasting.metrics as fcMetric
import torch.optim as optim
import matplotlib.pyplot as plt
import os

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_head):
        super().__init__()
        hidden_dim = int(hidden_dim) # For Re-formatting for b optimization space
        num_head = int(num_head)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_head, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.attn(x, x, x) # Arguments respectively (Q, K, V)
        out, _ = self.lstm2(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

data, scaler, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=1)
input_dim = 1 # All closed values
epoch_num = 100 # Proper epoch num found with BO, rounded to 100

def train_model(mdl, epoch_num, optimzr, cr, lossLst, isScientific=False, period=10):
  start = time.time()
  for epoch in range(epoch_num):
    mdl.train()
    for xb, yb in train_loader:
      optimzr.zero_grad()
      out = mdl(xb)
      loss = cr(out, yb)
      loss.backward()
      optimzr.step()
    lossLst.append(loss.item())

    if epoch % period == 0 or epoch == epoch_num-1:
      print(f"Epoch: {epoch}, Loss: {loss.item():.4f}" if not isScientific else f"Epoch: {epoch}, Loss: {loss.item():.2e}")
  end = time.time()
  return lossLst, end-start

model = LSTMAttentionModel(input_dim, hidden_dim=209, output_dim=pred_len, num_head=1)

lr = .001158
criterion = fcMetric.SMAPE()
optimizer = optim.Adam(model.parameters(), lr=lr)

if not os.path.exists("Resources/modelS.pt"):
    lossLst, totalt = train_model(model, epoch_num, optimizer, criterion, [], isScientific=False, period=10)

    plt.plot(lossLst, color="red")
    plt.xlabel("Training loss (SMAPE)", color="White")
    plt.ylabel("Epoch", color="White")
    plt.title("Single Prediction Model Training Loss", color="White")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)  # Making spines thicker
        spine.set_edgecolor('white')

    fig.set_size_inches(15, 6)
    fig.patch.set_facecolor('black')
    if not os.path.exists("Resources/Images/SingModelTrain.png"):
        fig.savefig("Resources/Images/SingModelTrain.png", dpi=300)

    plt.show()
    print(f"Total time required: {totalt:.4f} seconds")
    # Total time required: 1063.8598 seconds (Depends on hardware, expected less but the cpu is quite busy at training) 292.5236

    torch.save(model.state_dict(), "Resources/modelS.pt")