# optimization-related helper functions
import torch 
torch.set_default_dtype(torch.float32)

def optimal_decision(y: torch.Tensor, 
                     Zo: torch.Tensor, 
                     Zd: torch.Tensor, 
                     Zr: torch.Tensor, 
                     c_h: float = 1, c_p: float = 5, c_f: float = 25):
    """
    y:          [bs], integer
    Zo:         [no]
    Zd:         [nd], sorted ascending
    Zr:         [nr]
    return:
        z_opt:  [bs, 3]
    """
    device = y.device
    dtype = y.dtype if y.is_floating_point() else torch.float32
    bs = y.shape[0]
    no = len(Zo)
    nd = len(Zd)
    nr = len(Zr)
    # ----- all Zo x Zd x Zr combinations -----
    y_ = y[:, None, None, None]           # [bs, 1, 1, 1]
    zo = Zo[None, :, None, None]          # [1, no, 1, 1]
    zd = Zd[None, None, :, None]          # [1, 1, nd, 1]
    zr = Zr[None, None, None, :]          # [1, 1, 1, nr]
    # ----- feasibility: zo <= zr, zd <= zr -----
    feasible = (zo <= zr) & (zd <= zr)    # [1, no, nd, nr]
    # ----- lead time scenarios -----
    L = torch.tensor([3., 4., 5., 6., 7.], device=device, dtype=dtype)
    y5 = y_[..., None]                    # [bs, 1, 1, 1, 1]
    zo5 = zo[..., None]                   # [1, no, 1, 1, 1]
    zd5 = zd[..., None]                   # [1, 1, nd, 1, 1]
    zr5 = zr[..., None]                   # [1, 1, 1, nr, 1]
    arrival = zo5 + L                     # [1, no, 1, 1, 5]
    T = torch.maximum(zr5, arrival)        # [1, no, 1, nr, 5]
    # ----- (a) holding cost -----
    holding = c_h * (zr5 - arrival).clamp_min(0)
    # ----- (b) opportunity cost -----
    opportunity = c_p * (torch.maximum(y5, T) - zd5).clamp_min(0)
    # ----- (c) failure cost -----
    failure = c_f * (T - y5).clamp_min(0) * (y5 < zd5).to(dtype)
    # ----- expected cost over lead time -----
    expected_cost = (holding + opportunity + failure).mean(dim=-1)   # [bs, no, nd, nr]
    expected_cost = expected_cost.masked_fill(~feasible, torch.inf)
    # ----- joint argmin -----
    flat_idx = expected_cost.flatten(1).argmin(dim=1)
    idx_o = flat_idx // (nd * nr)
    remainder = flat_idx % (nd * nr)
    idx_d = remainder // nr
    idx_r = remainder % nr
    return torch.stack([Zo[idx_o], Zd[idx_d], Zr[idx_r]], dim=1)

def evaluate_decision(Z: torch.Tensor, y: torch.Tensor, 
                      c_h: float = 1, c_p: float = 5, c_f: float = 25):
    """
    Z: [bs, 3]
    y: [bs]
    return:
        expected_cost: [bs]
    """
    device = Z.device
    dtype = Z.dtype if Z.is_floating_point() else torch.float32
    zo = Z[:, 0:1]
    zd = Z[:, 1:2]
    zr = Z[:, 2:3]
    y_ = y[:, None]
    L = torch.tensor([3., 4., 5., 6., 7.], device=device, dtype=dtype)[None, :]
    arrival = zo + L
    T = torch.maximum(arrival, zr)
    holding = c_h * (zr - arrival).clamp_min(0)
    opportunity = c_p * (torch.maximum(y_, T) - zd).clamp_min(0)
    failure = c_f * (T - y_).clamp_min(0) * (y_ < zd).to(dtype)
    expected_cost = (holding + opportunity + failure).mean(dim=1)
    return expected_cost

def cendiff_grad(
    RUL: torch.Tensor,
    y_true: torch.Tensor,
    Zo: torch.Tensor,
    Zd: torch.Tensor,
    Zr: torch.Tensor,
    H: int = 125,
    c_h: float = 1,
    c_p: float = 5,
    c_f: float = 25,
):
    """
    Central-/one-sided-difference surrogate gradient.
    RUL:
        [bs], scalar predicted RUL, possibly continuous.
    y_true:
        [bs], realized RUL.
    H:
        number of discrete RUL support points / intervals.
    Convention:
        I_h = [h-1, h), h = 1,...,H-1
        I_H = [H-1, +inf)
    Returns:
        grad: [bs]
    """
    # ---------------------------------------------------------
    # 1. Map scalar prediction to RUL-support index kappa(RUL)
    #    h is 1-based, consistent with the manuscript.
    # ---------------------------------------------------------
    h = torch.floor(RUL).long() + 1
    h = h.clamp(min=1, max=H)
    # ---------------------------------------------------------
    # 2. Neighboring support indices
    # ---------------------------------------------------------
    h_minus = (h - 1).clamp(min=1)
    h_plus  = (h + 1).clamp(max=H)
    # support value associated with event h is h - 1
    y_minus = (h_minus - 1).to(dtype=RUL.dtype)
    y_plus  = (h_plus  - 1).to(dtype=RUL.dtype)
    # ---------------------------------------------------------
    # 3. Optimal decisions under the neighboring
    #    degenerate RLDs
    # ---------------------------------------------------------
    Z_minus = optimal_decision(y_minus, Zo, Zd, Zr, c_h=c_h, c_p=c_p, c_f=c_f)
    Z_plus = optimal_decision(y_plus, Zo, Zd, Zr, c_h=c_h, c_p=c_p, c_f=c_f)
    # ---------------------------------------------------------
    # 4. Evaluate both decisions under the realized RUL
    # ---------------------------------------------------------
    C_minus = evaluate_decision( Z_minus, y_true, c_h=c_h, c_p=c_p, c_f=c_f )
    C_plus = evaluate_decision( Z_plus, y_true, c_h=c_h, c_p=c_p, c_f=c_f )
    # ---------------------------------------------------------
    # 5. Central difference in the interior;
    #    one-sided difference at the boundaries
    # ---------------------------------------------------------
    denom = (h_plus - h_minus).to(dtype=C_plus.dtype)
    grad = (C_plus - C_minus) / denom
    return grad

def samples_to_prob(pred: torch.Tensor, support: torch.Tensor):
    """
    pred:    [bs, n_samples]
    support: [H], consecutive integers

    return:
        prob: [bs, H]
    """
    bs, n = pred.shape
    support = support.to(pred.device)
    H = support.numel()
    s_min = support[0]
    s_max = support[-1]
    pred = pred.clamp(min=s_min, max=s_max)
    idx = (pred - s_min).long()
    prob = torch.zeros(bs, H, device=pred.device, dtype=torch.float32)
    prob.scatter_add_(dim=1, index=idx, src=torch.ones_like(pred, dtype=prob.dtype))
    prob /= n
    return prob

def stochastic_decision(support: torch.Tensor, prob: torch.Tensor, 
                        Zo: torch.Tensor, Zd: torch.Tensor, Zr: torch.Tensor, 
                        c_h: float = 1, c_p: float = 5, c_f: float = 25):
    """
    Exact batch optimization under a discrete predictive distribution.
    support: [H]
    prob:    [bs, H]
    Zo:      [no]
    Zd:      [nd]
    Zr:      [nr]
    return:
        Z_opt: [bs, 3]
    Objective:
        E_{Y,L}[
            c_h [Zr - (Zo + L)]^+
            + c_p [max{Y, Zr, Zo + L} - Zd]^+
            + c_f [max{Zr, Zo + L} - Y]^+ 1{Y < Zd}
        ]
    Feasibility:
        Zo <= Zr, Zd <= Zr
    L ~ Uniform{3,4,5,6,7}
    """

    device = prob.device
    dtype = prob.dtype if prob.dtype in (torch.float32, torch.float64) else torch.float32

    if prob.ndim != 2:
        raise ValueError("prob must have shape [bs, H].")
    if prob.shape[1] != support.numel():
        raise ValueError("prob.shape[1] must equal support.numel().")
    if support.numel() == 0 or Zo.numel() == 0 or Zd.numel() == 0 or Zr.numel() == 0:
        raise ValueError("support, Zo, Zd and Zr must be non-empty.")
    if torch.any(prob < 0):
        raise ValueError("Probabilities must be nonnegative.")

    # ----- sort support and decision sets -----
    support, s_idx = torch.sort(support)
    prob = prob[:, s_idx]
    Zo = torch.sort(Zo).values
    Zd = torch.sort(Zd).values
    Zr = torch.sort(Zr).values
    bs, H = prob.shape
    no = Zo.numel()
    nd = Zd.numel()
    nr = Zr.numel()
    # ----- normalize probability -----
    prob_sum = prob.sum(dim=1, keepdim=True)
    if torch.any(prob_sum <= 0):
        raise ValueError("Each probability vector must have positive total probability.")
    prob = prob / prob_sum
    # ----- prefix probability and first moment -----
    zero = torch.zeros(bs, 1, device=device, dtype=dtype)
    F = torch.cat([zero, prob.cumsum(dim=1)], dim=1)
    M = torch.cat([zero, (prob * support[None, :]).cumsum(dim=1)], dim=1)
    mu = M[:, -1]
    # ----- P(Y < Zd) and E[Y 1{Y < Zd}] -----
    idx_lt_d = torch.searchsorted(support, Zd, right=False)
    F_lt_d = F[:, idx_lt_d]
    M_lt_d = M[:, idx_lt_d]
    # ----- expectation over lead time -----
    L = torch.tensor([3., 4., 5., 6., 7.], device=device, dtype=dtype)
    T_bar = torch.zeros(no, nr, device=device, dtype=dtype)
    holding_bar = torch.zeros(no, nr, device=device, dtype=dtype)
    tail_bar = torch.zeros(bs, no, nr, device=device, dtype=dtype)
    for ell in L:
        arrival = Zo[:, None] + ell
        T = torch.maximum(arrival, Zr[None, :])
        idx_le_t = torch.searchsorted(support, T, right=True)
        F_le_t = F[:, idx_le_t]
        M_le_t = M[:, idx_le_t]
        tail = mu[:, None, None] - M_le_t - T[None, :, :] * (1. - F_le_t)
        T_bar += T
        holding_bar += (Zr[None, :] - arrival).clamp_min(0)
        tail_bar += tail
    T_bar = T_bar / L.numel()
    holding_bar = holding_bar / L.numel()
    tail_bar = tail_bar / L.numel()
    # ----- terms independent of Zd -----
    base = c_h * holding_bar[None, :, :] + c_p * (T_bar[None, :, :] + tail_bar)
    # ----- exact minimization over Zd x Zo x Zr -----
    best_cost = torch.full((bs,), torch.inf, device=device, dtype=dtype)
    best_o = torch.zeros(bs, device=device, dtype=torch.long)
    best_d = torch.zeros(bs, device=device, dtype=torch.long)
    best_r = torch.zeros(bs, device=device, dtype=torch.long)
    target_elements = 8_000_000
    chunk_d = max(1, min(nd, target_elements // max(bs * no * nr, 1)))
    for start in range(0, nd, chunk_d):
        end = min(start + chunk_d, nd)
        d = Zd[start:end]
        Fd = F_lt_d[:, start:end]
        Md = M_lt_d[:, start:end]
        cost = base[:, None, :, :] - c_p * d[None, :, None, None] + c_f * (Fd[:, :, None, None] * T_bar[None, None, :, :] - Md[:, :, None, None])
        feasible = ((Zo[None, None, :, None] <= Zr[None, None, None, :]) & 
                    (d[None, :, None, None] <= Zr[None, None, None, :]) )
        cost = cost.masked_fill(~feasible, torch.inf)
        chunk_cost, flat_idx = cost.flatten(1).min(dim=1)
        idx_d_local = flat_idx // (no * nr)
        remainder = flat_idx % (no * nr)
        idx_o_local = remainder // nr
        idx_r_local = remainder % nr
        better = chunk_cost < best_cost
        best_cost = torch.where(better, chunk_cost, best_cost)
        best_d = torch.where(better, idx_d_local + start, best_d)
        best_o = torch.where(better, idx_o_local, best_o)
        best_r = torch.where(better, idx_r_local, best_r)
    Z_opt = torch.stack([Zo[best_o], Zd[best_d], Zr[best_r]], dim=1)

    return Z_opt

