import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_forecasting.metrics as fcMetric
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from torchinfo import summary
from PreProcessMinMax import preProcess

# Without ABBA representation preprocessing
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

space = [
    Integer(32, 256, name="hidden_dim"),
    Integer(1, 8, name="num_head"),
    Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
    Integer(10, 30, name="epoch_num")
]

space_epoch = [ # For declaring epoch number on high numbers
    Integer(25, 250, name="epoch_num")
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@use_named_args(space) 
def objective_fc(hidden_dim, num_head, lr, epoch_num):

    if int(hidden_dim) % int(num_head) != 0:
        return 1e6

    data, scaler, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=20)
    input_dim=1
    output_dim=20 # A month long
    model = LSTMAttentionModel(input_dim, hidden_dim, output_dim, num_head).to(device)
    criterion = fcMetric.SMAPE()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epoch_num):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.squeeze(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = .0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb.squeeze(-1)).item()
    val_loss /= len(test_loader)

    return val_loss

@use_named_args(space_epoch) 
def objective_findEpoch(epoch_num):

    hidden_dim = 132
    lr = 0.000829
    num_head = 2

    data, scaler, (block_size, split, pred_len), (train_loader, test_loader) = preProcess("AAPL", "1d", block_size=32, split=.8, pred_length=20)
    input_dim=1
    output_dim=20 # A month long
    model = LSTMAttentionModel(input_dim, hidden_dim, output_dim, num_head).to(device)
    criterion = fcMetric.SMAPE()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epoch_num):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.squeeze(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = .0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb.squeeze(-1)).item()
    val_loss /= len(test_loader)

    return val_loss