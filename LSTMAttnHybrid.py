import torch.nn as nn
import torch
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import pytorch_forecasting.metrics as fcMetric
from PreProcessingABBA import preprocess
import torch.optim as optim


class LSTMAttentionHybridModel_ABBA(nn.Module):
  def __init__(self, embed_dim, hidden_dim, output_dim, vocab_size, block_size, dense_units=100, num_lstm_layers=1):
    super().__init__()
    hidden_dim = int(hidden_dim)
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)
    self.multiheadAttn = nn.MultiheadAttention(embed_dim, num_heads=embed_dim, batch_first=True)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(block_size*(1+embed_dim+hidden_dim), dense_units)
    self.fc2 = nn.Linear(dense_units, output_dim)

  def forward(self, x_ids):
    x_raw = x_ids.float() # (B, L)
    x = self.embedding(x_ids) # (B, L, E)
    lstm_out, _ = self.lstm(x) # (B, L, H)
    attn_out, _ = self.multiheadAttn(x, x, x) # (B, L, L)
    lstm_out, attn_out = self.flatten(lstm_out), self.flatten(attn_out)

    # print(f"{x_raw.shape}, {lstm_out.shape}, {attn_out.shape}")

    flat_x = torch.cat([lstm_out, attn_out, x_raw], dim=-1)
    densed = self.fc1(flat_x)
    out = self.fc2(densed)
    return out
  
space = [
            Integer(32, 256, name="hidden_dim"),
            Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
            Integer(10, 30, name="epoch_num"),
            Real(1e-2, 10, prior="log-uniform", name="tol")
        ]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@use_named_args(space) 
def objective_fcA(hidden_dim, lr, epoch_num, tol):

    _, (block_size, _, _), (train_loader, test_loader), (string, parameters, pieces), vocab_size = preprocess("AAPL", "1d", block_size=32, split=.8, pred_length=20, tol=tol)

    input_dim=1
    output_dim=20 # A month long
    model = LSTMAttentionHybridModel_ABBA(input_dim, hidden_dim, output_dim, vocab_size, block_size=block_size).to(device)
    criterion = fcMetric.SMAPE()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epoch_num):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = .0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb).item()
    val_loss /= len(test_loader)

    return val_loss