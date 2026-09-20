
import torch
import argparse
import utlis as ut
from time import time
from methods import ETO, DOT, IEO
torch.set_default_dtype(torch.float32)
# switch your device here.
device = "cuda:0"
# device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--machine", type=str,      required=True, default="FD001")
parser.add_argument("--factor",  type=float,    required=True, default=5.0)
parser.add_argument("--trials",  type=int,      required=True, default=10)
args = parser.parse_args()

# across experiments
factor = args.factor    # 2, 5, 10, 20
machine = args.machine  # FD001,...,FD004
nseed  = args.trials    # repeat how many times

c_h = 1.; c_p = 5.; c_f = factor * c_p
Z_bar = 100
H = 125
Δr = 5

Zo = torch.arange(0, Z_bar,         device=device)
Zd = torch.arange(0, Z_bar,         device=device)
Zr = torch.arange(Δr,Z_bar+Δr, Δr,  device=device)
support = torch.arange(0, H,        device=device)
problem = [Zo, Zd, Zr, support, c_h, c_p, c_f]
report = torch.zeros([nseed, 3, 7])

print("Current Experiment: Dataset [%s] Factor [%.0f] Repeat [%02d] times " % (machine, factor, nseed))

for seed in range(nseed):
    x_train, y_train, x_valid, y_valid, x_check, y_check = ut.load_data("./data/X_Y_%s.csv" % machine, L=30, R=Z_bar-1, seed=seed)
    lrset = [1e-4, 2e-4, 5e-4, 1e-3]

    md = 0
    t1 = time()
    ETOmodel = ETO.Tuning(x_train, y_train, x_valid, y_valid, problem, lrset, device)
    t2 = time()
    MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ = ETO.Apply(x_check, y_check, ETOmodel, problem, device)
    report[seed, md, 0] = MAE;   report[seed, md, 1] = CRPS_1; report[seed, md, 2] = CRPS_2
    report[seed, md, 3] = C_AVG; report[seed, md, 4] = L_FREQ; report[seed, md, 5] = R_FREQ
    report[seed, md, 6] = t2 - t1
    print("Run: %02d, ETO %.1f %.1f %.1f %.1f %.3f %.3f %.1f" % (seed+1, MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ, t2-t1))

    md = 1
    t1 = time()
    DOTmodel = DOT.Tuning(x_train, y_train, x_valid, y_valid, problem, lrset, device)
    t2 = time()
    MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ = ETO.Apply(x_check, y_check, DOTmodel, problem, device)
    report[seed, md, 0] = MAE;   report[seed, md, 1] = CRPS_1; report[seed, md, 2] = CRPS_2
    report[seed, md, 3] = C_AVG; report[seed, md, 4] = L_FREQ; report[seed, md, 5] = R_FREQ
    report[seed, md, 6] = t2 - t1
    print("Run: %02d, DOT %.1f %.1f %.1f %.1f %.3f %.3f %.1f" % (seed+1, MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ, t2-t1))

    md = 2
    t1 = time()
    IEOmodel = IEO.Tuning(x_train, y_train, x_valid, y_valid, ETOmodel, problem, lrset, device)
    t2 = time()
    MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ = ETO.Apply(x_check, y_check, IEOmodel, problem, device)
    report[seed, md, 0] = MAE;   report[seed, md, 1] = CRPS_1; report[seed, md, 2] = CRPS_2
    report[seed, md, 3] = C_AVG; report[seed, md, 4] = L_FREQ; report[seed, md, 5] = R_FREQ
    report[seed, md, 6] = t2 - t1
    print("Run: %02d, IEO %.1f %.1f %.1f %.1f %.3f %.3f %.1f" % (seed+1, MAE, CRPS_1, CRPS_2, C_AVG, L_FREQ, R_FREQ, t2-t1))
    
filename = "./results/%s_ratio=%02d.pth" % (machine, factor)

torch.save(report, filename)

