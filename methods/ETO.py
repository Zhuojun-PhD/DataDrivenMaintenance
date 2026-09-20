# ============================================================== #
# Code for Estimate-then-Optimize approach
# including training, validation, and testing (we call checking)
# ============================================================== #
import opt
import torch 
import utlis as ut
from torch.utils.data import TensorDataset, DataLoader
torch.set_default_dtype(torch.float32)

bs = 100
patience = 10

def Train_And_Validate(x_train: torch.Tensor,
                       y_train: torch.Tensor,
                       x_valid: torch.Tensor,
                       y_valid: torch.Tensor,
                       lr: float,
                       device: str = "cuda:0"):
    
    worker = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    winner = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    
    optimizer = torch.optim.Adam(worker.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
    
    x_valid_cuda = x_valid.to(device)
    y_valid_cuda = y_valid.to(device)
    
    count = 0; bester = torch.inf
    
    for epoch in range(1_000):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_fct = worker(x_batch)
            loss = criterion(y_fct, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_valid_fct = worker.mc_forward(x_valid_cuda, 1000).floor()
            score = ((y_valid_fct - y_valid_cuda[:, None]) ** 2).mean()
        if score < bester:
            bester = score
            count = 0
            winner.load_state_dict(worker.state_dict())
        else:
            count += 1
        if count == patience:
            break

    return winner, bester

def Tuning(x_train: torch.Tensor,
           y_train: torch.Tensor,
           x_valid: torch.Tensor,
           y_valid: torch.Tensor,
           problem: list,
           lr_list: list,
           device: str = "cuda:0"):
    ''' 
    This function tunes hyperparameters for the model including
    learning rate and weight_decay
    
    For ETO, we use CRPS score to choose hyperparameters
    For DOT, we use decision quality to choose hyperparameters
    '''
    # For ETO approach, try different combinations of learning rate and weight decay
    ETO_best_score = torch.inf
    ETOmodel = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    for lr in lr_list:
        model, CRPSscore = Train_And_Validate(x_train, y_train, x_valid, y_valid, lr, device)
        if CRPSscore < ETO_best_score:
            ETO_best_score = CRPSscore
            ETOmodel.load_state_dict(model.state_dict())
    return ETOmodel

def Apply(x_check: torch.Tensor, 
          y_check: torch.Tensor,
          model: ut.LSTM,
          problem: list,
          device: str = "cuda:0"):
    ''' 
    This function applies the tuned and trained model to the testing (check) set
    and report different metrics including average MAE, CRPS, Decision Quality.
    '''
    Zo, Zd, Zr, support, c_h, c_p, c_f = problem
    model.requires_grad_(False)
    model = model.to(device)

    # 1. make MC prediction
    x_gpu = x_check.to(device)
    y_gpu = y_check.to(device)
    y_fct = model.mc_forward(x_gpu, 1000)

    # 2. compute MAE and CRPS
    mae = torch.mean(torch.abs(y_fct - y_gpu[:, None])).item()
    CRPS_left, CRPS_right = ut.CRPS(y_fct, y_gpu)
    CRPS_1 = CRPS_left.item()
    CRPS_2 = CRPS_right.item()

    # 3. compute decision quality
    p_fct = opt.samples_to_prob(y_fct.floor(), support)
    z_fct = opt.stochastic_decision(support, p_fct, Zo, Zd, Zr, c_h, c_p, c_f)
    v_fct = opt.evaluate_decision(z_fct, y_gpu, c_h, c_p, c_f)
    c_avg = v_fct.mean().item()

    # 5. compute early/later downtime frequency
    early_frq = (z_fct[:, 1] <= y_gpu).float().mean()
    later_frq = (z_fct[:, 1] > y_gpu).float().mean()

    return mae, CRPS_1, CRPS_2, \
           c_avg, early_frq.item(), later_frq.item() # 6 quantities