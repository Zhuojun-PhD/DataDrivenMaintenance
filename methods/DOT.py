# ======================================================================== #
# Code for Decision-Oriented hyperparameter Tuning (DOT)
# including training, validation, and testing (we call checking)
# the core idea is during training, we use the decision-quality
# of the trained model to early-stop (on the validation set)
# also, we use cross-hyperparameter decision quality on the validation set
# to determine the hyperparameters.
# ======================================================================== #
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
                       problem: list, 
                       lr: float,
                       device: str = "cuda:0"):
    
    Zo, Zd, Zr, support, c_h, c_p, c_f = problem
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
            
        worker.train()
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
           problem: list,
           lr_list: list,
           device: str = "cuda:0"):
    ''' 
    This function tunes hyperparameters for the model including
    learning rate and weight_decay using the validation set with the decision-oriented criterion
    '''
    # For DOT approach, try different combinations of learning rate and weight decay
    best_score = torch.inf
    DOTmodel = ut.LSTM(x_train.shape[-1], dropout=0.20).to(device)
    for lr in lr_list:
        model, score = Train_And_Validate(x_train, y_train, x_valid, y_valid, problem, lr, device)
        if score < best_score:
            best_score = score
            DOTmodel.load_state_dict(model.state_dict())
    return DOTmodel
