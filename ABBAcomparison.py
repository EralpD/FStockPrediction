from PreProcessingABBA import preprocess
from PreProcessMinMax import preProcess
from LSTMAttnSequential import LSTMAttentionModel
import pytorch_forecasting.metrics as fcMetric
import torch.optim as optim
import torch.nn as nn
import torch
import time

class LSTMAttentionModel_ABBA(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, vocab_size, num_head):
        super().__init__()
        hidden_dim = int(hidden_dim) # For Re-formatting for b optimization space
        num_head = int(num_head)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_head, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_ids):
        x = self.embedding(x_ids)
        x, _ = self.lstm1(x)
        x, _ = self.attn(x, x, x) # Arguments respectively (Q, K, V)
        out, _ = self.lstm2(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class LSTMAttentionHybridModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, block_size, dense_units=100, num_lstm_layers=1, num_heads=0):
    super().__init__()
    if num_heads == 0:
      num_heads = block_size
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)
    self.multiheadAttn = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(block_size*(2*input_dim+hidden_dim), dense_units)
    self.fc2 = nn.Linear(dense_units, output_dim)

  def forward(self, x):
    lstm_out, _ = self.lstm(x) # (B, L, H)
    attn_out, _ = self.multiheadAttn(x, x, x) # (B, L, I)
    lstm_out, attn_out = self.flatten(lstm_out), self.flatten(attn_out)

    # print(f"{x_raw.shape}, {lstm_out.shape}, {attn_out.shape}")

    x = x.squeeze(-1)
    flat_x = torch.cat([lstm_out, attn_out, x], dim=-1)
    densed = self.fc1(flat_x)
    out = self.fc2(densed)
    return out
    
def train_model(mdl, epoch_num, optimzr, cr, lossLst, isScientific=False, period=10):
  start = time.time()
  for epoch in range(epoch_num):
    mdl.train()
    for xb, yb in train_loader:
      optimzr.zero_grad()
      out = mdl(xb)
      loss = cr(out, yb.squeeze(-1))
      loss.backward()
      optimzr.step()
    lossLst.append(loss.item())

    if epoch % period == 0 or epoch == epoch_num-1:
      print(f"Epoch: {epoch}, Loss: {loss.item():.4f}" if not isScientific else f"Epoch: {epoch}, Loss: {loss.item():.2e}")
  
  end = time.time()
  total = end - start
  print(f"Time spent without ABBA preprocessing: {total:.2f} seconds")

  return lossLst

def train_model_ABBA(mdl, epoch_num, optimzr, cr, lossLst, isScientific=False, period=10):
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
  total = end - start
  print(f"Time spent without ABBA preprocessing: {total:.2f} seconds")

  return lossLst

# Without ABBA part
data, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=20)
model = LSTMAttentionModel(1, 209, pred_len, num_head=1)

criterion = fcMetric.SMAPE()
optimizer = optim.Adam(model.parameters(), lr=.000829)

lossLst = train_model(model, 100, optimizer, criterion, [], isScientific=False, period=10)
avg = sum(lossLst) / len(lossLst) if len(lossLst) > 1 else sum(lossLst)
print(f"Average loss score: {avg:.4f}", )

# ABBA part (tol part gain from second model's BO process)
# data, (block_size, split, pred_len), (train_loader, test_loader), (string, parameters, pieces), vocab_size = preprocess("AAPL", "1d", block_size=32, split=.8, pred_length=20, tol=9.0455)
# model = LSTMAttentionModel_ABBA(8, 209, pred_len, vocab_size=vocab_size,num_head=1) # 8 is selected randomized, will be selected manually
# criterion = fcMetric.SMAPE()
# optimizer = optim.Adam(model.parameters(), lr=.001158)

# lossLst = train_model_ABBA(model, 100, optimizer, criterion, [], isScientific=False, period=10)
# avg = sum(lossLst) / len(lossLst) if len(lossLst) > 1 else sum(lossLst)
# print(f"Average loss score: {avg:.4f}", )

# Extra without ABBA model

data, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=20)
model = LSTMAttentionHybridModel(1, 209, pred_len, block_size, num_heads=1)

criterion = fcMetric.SMAPE()
optimizer = optim.Adam(model.parameters(), lr=0.000129)

lossLst = train_model(model, 100, optimizer, criterion, [], isScientific=False, period=10)
avg = sum(lossLst) / len(lossLst) if len(lossLst) > 1 else sum(lossLst)
print(f"Average loss score: {avg:.4f}", )
