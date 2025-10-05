from PreProcessMinMax import preProcess
from LSTMAttnSequential import LSTMAttentionModel
import pytorch_forecasting.metrics as fcMetric
import torch.optim as optim
import time

data, scaler, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=20)

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

# Best parameters:
# hidden_dim=209, num_head=1, lr=0.001158, epoch_num=16
model = LSTMAttentionModel(1, 209, pred_len, num_head=1)

criterion = fcMetric.SMAPE()
optimizer = optim.Adam(model.parameters(), lr=.001158)

lossLst = train_model(model, 500, optimizer, criterion, [], isScientific=False, period=5) # Finding most proper epoch number with manually
epoch_num = lossLst.index(min(lossLst))
print("Must Proper Epoch number: ", epoch_num)
# Proper epoch number: ~102 (This is valid because best epoch_number from BO,16 tried many time and got most likely same results.)