import torch
import numpy as np
import pandas as pd
from torch import nn
torch.set_default_dtype(torch.float32)

def load_by_cell(csv_path: str) -> list[np.ndarray]:
    df = pd.read_csv(csv_path)
    feature_cols = [ "obs_%d" % (i) for i in range(1,22) ]
    arrays = [ group[feature_cols + ["RUL"]].to_numpy() for _, group in df.groupby("id", sort=False) ]
    return arrays

def make_windows( block: np.ndarray, window_size: int ) -> tuple[np.ndarray, np.ndarray]:
    block = np.asarray(block)
    features = block[:, :-1]
    targets = block[:, -1]
    num_cycles, num_features = features.shape
    num_windows = num_cycles - window_size + 1
    start_indices = np.arange(num_windows)[:, None]
    window_offsets = np.arange(window_size)[None, :]
    window_indices = start_indices + window_offsets
    X = features[window_indices]
    Y = targets[window_size - 1:].copy()
    return X, Y

def load_data(path: str, L: int, R: float, seed: int = 0):
    data = load_by_cell(path)
    # randomly split training/validation/testing
    # use random seed for reproducibility
    np.random.seed(seed)
    machineset = np.random.permutation(100)
    train_set = machineset[:70]
    valid_set = machineset[70:80]
    check_set = machineset[80:]
        
    X_list, y_list = zip(*(make_windows(data[i], L) for i in train_set))
    X_train = np.concatenate(X_list, axis=0, dtype=np.float32)
    y_train = np.concatenate(y_list, axis=0, dtype=np.float32)

    X_list, y_list = zip(*(make_windows(data[i], L) for i in valid_set))
    X_valid = np.concatenate(X_list, axis=0, dtype=np.float32)
    y_valid = np.concatenate(y_list, axis=0, dtype=np.float32)

    X_list, y_list = zip(*(make_windows(data[i], L) for i in check_set))
    X_check = np.concatenate(X_list, axis=0, dtype=np.float32)
    y_check = np.concatenate(y_list, axis=0, dtype=np.float32)

    # Keep only samples with RUL below or equal to the cutoff
    train_sample_mask = y_train <= R
    valid_sample_mask = y_valid <= R
    check_sample_mask = y_check <= R

    X_train = X_train[train_sample_mask]
    y_train = y_train[train_sample_mask]
    X_valid = X_valid[valid_sample_mask]
    y_valid = y_valid[valid_sample_mask]
    X_check = X_check[check_sample_mask]
    y_check = y_check[check_sample_mask]

    feature_mask = X_train[:, -1, :].std(axis=0, ddof=1) != 0.
    X_train = X_train[:, :, feature_mask]
    X_valid = X_valid[:, :, feature_mask]
    X_check = X_check[:, :, feature_mask]
    # Remove constant or nearly constant features
    feature_mask = X_train[:, -1, :].std(axis=0, ddof=1) > 1e-2
    X_train = X_train[:, :, feature_mask]
    X_valid = X_valid[:, :, feature_mask]
    X_check = X_check[:, :, feature_mask]

    # Normalize using training statistics
    feature_min = X_train[:, -1, :].min(axis=0)
    feature_max = X_train[:, -1, :].max(axis=0)

    X_train = 2 * ( X_train - feature_min ) / ( feature_max - feature_min ) - 1
    X_valid = 2 * ( X_valid - feature_min ) / ( feature_max - feature_min ) - 1
    X_check = 2 * ( X_check - feature_min ) / ( feature_max - feature_min ) - 1

    x_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).float()
    x_check = torch.from_numpy(X_check).float()
    y_check = torch.from_numpy(y_check).float()

    return x_train, y_train, x_valid, y_valid, x_check, y_check

def CRPS(pred: torch.Tensor, target: torch.Tensor):
    """
    pred:   [bs, M]
    target: [bs]

    return:
        crps_left:  scalar
        crps_right: scalar
    """
    target = target.to(device=pred.device, dtype=pred.dtype)
    pred, _ = torch.sort(pred, dim=1)
    bs, M = pred.shape
    y = target[:, None]
    # -------------------------
    # Left CRPS
    # -------------------------
    left_width = torch.clamp(torch.minimum(y, pred[:, 1:]) - pred[:, :-1], min=0)
    j = torch.arange(1, M, device=pred.device, dtype=pred.dtype)
    left = (left_width * (j / M) ** 2).sum(dim=1)
    left += torch.clamp(target - pred[:, -1], min=0)
    # -------------------------
    # Right CRPS
    # -------------------------
    right_width = torch.clamp(pred[:, 1:] - torch.maximum(y, pred[:, :-1]), min=0)
    right = (right_width * (1 - j / M) ** 2).sum(dim=1)
    right += torch.clamp(pred[:, 0] - target, min=0)
    return left.mean(), right.mean()

class LSTM(nn.Module):
    def __init__(self, dx: int, 
                 hidden_dim_lstm: int = 50, 
                 hidden_dim_fc: int = 100,  
                 dropout: float = 0.20):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=dx,
            hidden_size=hidden_dim_lstm,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_out_dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim_lstm, hidden_dim_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_fc, 1),
        )
        self.sps  = nn.Softplus()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # out: [bs, L, hidden_dim_lstm]
        last = out[:, -1, :]                # [bs, hidden_dim_lstm]
        last = self.lstm_out_dropout(last)  # dropout
        y = self.fc(last).squeeze(-1)       # [bs]
        return self.sps(y) * 100.           # simplify the learning scale

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                h = param.size(0) // 4
                param.data[h:2 * h].fill_(1.0)  # forget gate
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        
    @torch.no_grad()
    def mc_forward(
        self, x: torch.Tensor, M: int = 1000, 
        batch_size: int = 100, mc_chunk: int = 100) -> torch.Tensor:
        """
        x: [bs, L, dx]
        return: [bs, M]
        """
        bs, L, dx = x.shape
        y_all = torch.empty(bs, M, device=x.device)
        with torch.no_grad():
            for i in range(0, bs, batch_size):
                xb = x[i:i + batch_size]
                b = xb.shape[0]
                for j in range(0, M, mc_chunk):
                    c = min(mc_chunk, M - j)
                    x_mc = ( xb.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, L, dx) )
                    y_mc = self.forward(x_mc).reshape(b, c)
                    y_all[i:i + b, j:j + c] = y_mc
        return y_all