# ============================================================== #
# Code for Integrated Estimate-Optimize approach
# including training, validation, and testing (we call checking)
# we use validation set to early-stop when no decision quality 
# improvement, and use the same validation set to pick hyperparameters
# the training is composed with a warm-start from ETO.
# ============================================================== #
import opt
import torch 
import utlis as ut
from torch.utils.data import TensorDataset, DataLoader
torch.set_default_dtype(torch.float32)

H = 125
bs = 100
patience = 10

def Train_And_Validate(x_train: torch.Tensor,
                       y_train: torch.Tensor,
                       x_valid: torch.Tensor,
                       y_valid: torch.Tensor,
                       baseline: ut.LSTM,       # model inherited from ETO
                       problem: list, 
                       lr: float,
                       device: str = "cuda:0"):

    
    Zo, Zd, Zr, support, c_h, c_p, c_f = problem
    
    # create a worker and inherit from the ETO trained model
    worker = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    winner = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    
    worker.load_state_dict( baseline.to(device).state_dict() )
    optimizer = torch.optim.Adam(worker.parameters(), lr=lr)
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
    
    x_valid_cuda = x_valid.to(device)
    y_valid_cuda = y_valid.to(device)
    
    count = 0; bester = torch.inf
    
    for epoch in range(1_000):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_fct = worker(x_batch)
            RUL_fct = y_fct.detach().floor()
            grad  = opt.cendiff_grad(RUL_fct, y_batch, Zo, Zd, Zr, H, c_h, c_p, c_f)
            loss = (y_fct * grad).mean()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            y_valid_fct = worker.mc_forward(x_valid_cuda, 1000).floor()
            p_valid_fct = opt.samples_to_prob(y_valid_fct, support)
            z_fct = opt.stochastic_decision(support, p_valid_fct, Zo, Zd, Zr, c_h, c_p, c_f)
            v_fct = opt.evaluate_decision(z_fct, y_valid_cuda, c_h, c_p, c_f)
            score = v_fct.mean().item()
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
           baseline: ut.LSTM,
           problem: list,
           lr_list: list,
           device: str = "cuda:0"):
    ''' 
    This function tunes hyperparameters for the model including
    learning rate and weight_decay using the validation set with the decision-oriented criterion
    '''
    # For IEO approach, try different combinations of learning rate and weight decay
    best_score = torch.inf
    IEOmodel = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    for lr in lr_list:
        model, score = Train_And_Validate(x_train, y_train, x_valid, y_valid, baseline, problem, lr, device)
        if score < best_score:
            best_score = score
            IEOmodel.load_state_dict(model.state_dict())
    return IEOmodel
