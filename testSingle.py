import time
import torch
from PreProcessMinMax import preProcess
import torch.nn as nn
import pytorch_forecasting.metrics as fcMetric
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as pldates
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
print(pred_len)
input_dim = 1 # All closed values
epoch_num = 100 # Proper epoch num found with BO, rounded to 100

def test_model(mdl, cr, inverse=False):
  mdl.eval()
  val_losses = []
  all_preds = []

  with torch.no_grad():
    for xb, yb in test_loader:
      out = mdl(xb)
      loss = cr(out, yb)
      val_losses.append(loss.item())
      all_preds.extend(out)

  all_preds = torch.cat(all_preds)
  avg_val_loss = sum(val_losses) / len(val_losses)
  if inverse:
    all_preds = scaler.inverse_transform(all_preds.reshape(-1, 1))

  return all_preds, (val_losses, avg_val_loss)

model = LSTMAttentionModel(input_dim, 209, pred_len, num_head=1)
model.load_state_dict(torch.load("Resources/modelS.pt"))

lr = .001158
criterion = fcMetric.SMAPE()
optimizer = optim.Adam(model.parameters(), lr=lr)
all_preds, (val_losses, avg_val_loss) = test_model(model, criterion, inverse=True)

warning_text = "⚠️ Warning: Underfitting detected, please consider predictions enter a unrecognized state!"
threshold = .175

data_split = int(split*len(data))

plt.plot(data['Date'].iloc[:data_split], data['Close'].iloc[:data_split].values.reshape(-1, 1), color="red", label="Train")
plt.plot(data['Date'].iloc[data_split:], data['Close'].iloc[data_split:].values.reshape(-1, 1), color="blue", label="Validation", alpha=.7)
plt.plot(data['Date'].iloc[data_split:], all_preds, color="yellow", label="Prediction", alpha=.5)
plt.xlabel("Date", color="White")
plt.ylabel("Price (USD)", color="White")
plt.title("AAPL Daily Close Validation (2020–2025)", color="White")
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
for loss in val_losses:
   if loss >= threshold:
      fig.text(
        0.5, 1.09,        
        warning_text,
        transform=ax.transAxes,        # so it's relative to the axes
        ha='center', va='bottom',
        fontsize=12, color='yellow', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", edgecolor="yellow", alpha=0.8)
      )
      break

fig.set_size_inches(15, 6)
fig.patch.set_facecolor('black')
plt.legend(loc="upper left")
if not os.path.exists("Resources/Images/SingModelAAPLLoss.png"):
  fig.savefig("Resources/Images/SingModelAAPLLoss.png", dpi=300)

plt.show()

print()

plt.plot(range(0, len(val_losses)), val_losses, color="red")
plt.xlabel("Iteration", color="White")
plt.ylabel("Loss (SMAPE)", color="White")
plt.title("Validation loss", color="White")
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
if not os.path.exists("Resources/Images/sAAPLIterationLoss.png"):
  fig.savefig("Resources/Images/sAAPLIterationLoss.png", dpi=300)

plt.show()

data, scaler, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("NVDA", "1d", block_size=32, split=.8, pred_length=1)
all_preds, (val_losses, avg_val_loss) = test_model(model, criterion, inverse=True)

data_split = int(split*len(data))

plt.plot(data['Date'].iloc[:data_split], data['Close'].iloc[:data_split].values.reshape(-1, 1), color="red", label="Train")
plt.plot(data['Date'].iloc[data_split:], data['Close'].iloc[data_split:].values.reshape(-1, 1), color="blue", label="Validation", alpha=.7)
plt.plot(data['Date'].iloc[data_split:], all_preds, color="yellow", label="Prediction", alpha=.5)
plt.xlabel("Date", color="White")
plt.ylabel("Price (USD)", color="White")
plt.title("NVDA Daily Close Validation (2020–2025)", color="White")
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
for loss in val_losses:
   if loss >= threshold:
      fig.text(
        0.5, 1.09,  
        warning_text,
        transform=ax.transAxes,        # so it's relative to the axes
        ha='center', va='bottom',
        fontsize=12, color='yellow', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", edgecolor="yellow", alpha=0.8)
      )
      break

fig.set_size_inches(15, 6)
fig.patch.set_facecolor('black')
plt.legend(loc="upper left")
if not os.path.exists("Resources/Images/SingModelNVDALoss.png"):
  fig.savefig("Resources/Images/SingModelNVDALoss.png", dpi=300)

plt.show()

print()

plt.plot(range(0, len(val_losses)), val_losses, color="red")
plt.xlabel("Iteration", color="White")
plt.ylabel("Loss (SMAPE)", color="White")
plt.title("Validation loss", color="White")
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
if not os.path.exists("Resources/Images/sNVDAIterationLoss.png"):
  fig.savefig("Resources/Images/sNVDAIterationLoss.png", dpi=300)

# Seems that trained with one ticker state mostly same efficient with another tickers' prediction accuricies
plt.show()
