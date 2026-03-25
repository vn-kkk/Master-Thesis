################################################################################
## IMPORT ALL REQUIRED MODULES
################################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import itertools
import random
import time

import numpy as np
import torch.nn as nn

from numba import jit
from tqdm import tqdm  # Added for progress tracking

import tov_tide # Custom import

################################################################################
## GLOBAL CONSTANTS AND UNIT CONVERSION FACTORS
################################################################################
msun=147660                 # Solar mass in cm given by the formula G*M_sun/c^2

c=2.9979e10                 # speed of light in cm/s (CGS)
G=6.67408e-8                # gravitational constant in cm^3/gm/s^2 (CGS)

dkm = 1.3234e-06            # conversion of MeV/fm^3 to km^-2
dcgs = 1.78e12              # factor to convert from MeV/fm^3 to gm/cm^3
conv = 8.2601e-40           # dyn/cm^2 to km^-2
cgs1=1.7827e+12             # MeV/fm3 to gms/cm3
cgs2=1.6022e+33             # MeV/fm3 to dyne/cm2


################################################################################
## 4 PP LOW-DENSITY (CRUST PARAMETERS)
################################################################################
# Low density Polytropic exponents
GammaL_1 = 1.35692
GammaL_2 = 0.62223
GammaL_3 = 1.28733
GammaL_4 = 1.58425

# Low density Polytropic constants
KL_1 = 3.99874e-8 # * pow(Msun/Length**3, GammaL_1-1)
KL_2 = 5.32697e+1 # * pow(Msun/Length**3, GammaL_2-1)
KL_3 = 1.06186e-6 # * pow(Msun/Length**3, GammaL_3-1)
KL_4 = 6.80110e-9 # * pow(Msun/Length**3, GammaL_4-1)
# notice a missing c^2 in Ki values in Table II of Read et al. 2009

# Densities at the boundaries of the low density piecewise polytropes
rhoL_1 = 2.62789e12
rhoL_2 = 3.78358e11
rhoL_3 = 2.44034e7
rhoL_4 = 0.0

# Pressures at the boundaries of the low density piecewise polytropes
pL_1 = KL_1*rhoL_1**GammaL_1
pL_2 = KL_2*rhoL_2**GammaL_2
pL_3 = KL_3*rhoL_3**GammaL_3
pL_4 = 0.0

# The exact numbers are taken from a particular crust model/table.


################################################################################
## LOW-DENSITY ENERGY DENSITY, ALPHA AND BREAK DENSITIES
################################################################################
epsL_4 = 0.0
alphaL_4 = 0.0
epsL_3 = (1+alphaL_4)*rhoL_3 + KL_4/(GammaL_4 - 1.)*pow(rhoL_3, GammaL_4)
alphaL_3 = epsL_3/rhoL_3 - 1.0 - KL_3/(GammaL_3 - 1.)*pow(rhoL_3, GammaL_3 -1.0)
epsL_2 = (1+alphaL_3)*rhoL_2 + KL_3/(GammaL_3 - 1.)*pow(rhoL_2, GammaL_3)
alphaL_2 = epsL_2/rhoL_2 - 1.0 - KL_2/(GammaL_2 - 1.)*pow(rhoL_2, GammaL_2 -1.0)
epsL_1 = (1+alphaL_2)*rhoL_1 + KL_2/(GammaL_2 - 1.)*pow(rhoL_1, GammaL_2)
alphaL_1 = epsL_1/rhoL_1 - 1.0 - KL_1/(GammaL_1 - 1.)*pow(rhoL_1, GammaL_1 -1.0)

rho1 = pow(10,14.7) # Break Density 1
rho2 = pow(10,15.0) # Break Density 2

t_p=G/c**4
t_rho=G/c**2


################################################################################
## FORWARD EOS
################################################################################
def p_eps_of_rho(rho,logp,Gamma1,Gamma2,Gamma3):
    p1 = pow(10.0,logp)/c**2
    K1 = p1/pow(rho1,Gamma1)
    K2 = K1 * pow( rho1, Gamma1-Gamma2)
    K3 = K2 * pow( rho2, Gamma2-Gamma3)
    rho0 = pow(KL_1/K1,1.0/(Gamma1-GammaL_1)) 
    eps0 = (1.0+alphaL_1)*rho0 + KL_1/(GammaL_1-1.0)*pow(rho0,GammaL_1)
    alpha1 = eps0/rho0 - 1.0 - K1/(Gamma1 - 1.0)*pow(rho0, Gamma1 -1.0)
    eps1 = (1.0+alpha1)*rho1 + K1/(Gamma1 - 1.0)*pow(rho1, Gamma1)
    alpha2 = eps1/rho1 - 1.0 - K2/(Gamma2 - 1.0)*pow(rho1, Gamma2 -1.0)
    eps2 = (1.0+alpha2)*rho2 + K2/(Gamma2 - 1.0)*pow(rho2, Gamma2)
    alpha3 = eps2/rho2 - 1.0 - K3/(Gamma3 - 1.0)*pow(rho2, Gamma3 -1.0)
    if rho<rhoL_3:
        p = KL_4*pow(rho,GammaL_4)
        eps = (1.0+alphaL_4)*rho + KL_4/(GammaL_4-1.0)*pow(rho,GammaL_4)
    elif rhoL_3<= rho <rhoL_2:
        p = KL_3*pow(rho,GammaL_3)
        eps = (1.0+alphaL_3)*rho + KL_3/(GammaL_3-1.0)*pow(rho,GammaL_3)
    elif rhoL_2<= rho <rhoL_1:
        p = KL_2*pow(rho,GammaL_2)
        eps = (1.0+alphaL_2)*rho + KL_2/(GammaL_2-1.0)*pow(rho,GammaL_2)
    elif rhoL_1<= rho <rho0:
        p = KL_1*pow(rho,GammaL_1)
        eps = (1.0+alphaL_1)*rho + KL_1/(GammaL_1-1.0)*pow(rho,GammaL_1)
    elif rho0<= rho <rho1:
        p = K1*pow(rho,Gamma1)
        eps = (1.0+alpha1)*rho + K1/(Gamma1-1.0)*pow(rho,Gamma1)
    elif rho1<= rho <rho2:
        p = K2*pow(rho,Gamma2)
        eps = (1.0+alpha2)*rho + K2/(Gamma2-1.0)*pow(rho,Gamma2)
    else:
        p = K3*pow(rho,Gamma3)
        eps = (1.0+alpha3)*rho + K3/(Gamma3-1.0)*pow(rho,Gamma3)
    return p*c**2, eps*c**2


################################################################################
## INVERSE EOS
################################################################################
@jit(nopython=True)
def eps(p,logp,Gamma1,Gamma2,Gamma3):
    p1 = pow(10.0,logp)/c**2
    p*=1/c**2
    K1 = p1/pow(rho1,Gamma1)
    K2 = K1 * pow( rho1, Gamma1-Gamma2)
    K3 = K2 * pow( rho2, Gamma2-Gamma3)
    rho0 = pow(KL_1/K1,1.0/(Gamma1-GammaL_1))
    eps0 = (1.0+alphaL_1)*rho0 + KL_1/(GammaL_1-1.0)*pow(rho0,GammaL_1)
    alpha1 = eps0/rho0 - 1.0 - K1/(Gamma1 - 1.0)*pow(rho0, Gamma1 -1.0)
    eps1 = (1.0+alpha1)*rho1 + K1/(Gamma1 - 1.0)*pow(rho1, Gamma1)
    alpha2 = eps1/rho1 - 1.0 - K2/(Gamma2 - 1.0)*pow(rho1, Gamma2 -1.0)
    eps2 = (1.0+alpha2)*rho2 + K2/(Gamma2 - 1.0)*pow(rho2, Gamma2)
    alpha3 = eps2/rho2 - 1.0 - K3/(Gamma3 - 1.0)*pow(rho2, Gamma3 -1.0)
    p0 = K1*pow(rho0,Gamma1)
    p2 = K3*pow(rho2,Gamma3)
    if  p<pL_3:
        rho = pow(p/KL_4,1/GammaL_4)
        eps = (1.0+alphaL_4)*rho + KL_4/(GammaL_4-1.0)*pow(rho,GammaL_4)
    elif pL_3<= p <pL_2:
        rho = pow(p/KL_3,1/GammaL_3)
        eps = (1.0+alphaL_3)*rho + KL_3/(GammaL_3-1.0)*pow(rho,GammaL_3)
    elif pL_2<= p <pL_1:
        rho = pow(p/KL_2,1/GammaL_2)
        eps = (1.0+alphaL_2)*rho + KL_2/(GammaL_2-1.0)*pow(rho,GammaL_2)
    elif  pL_1<p <p0:
        rho = pow(p/KL_1,1/GammaL_1)
        eps = (1.0+alphaL_1)*rho + KL_1/(GammaL_1-1.0)*pow(rho,GammaL_1)
    elif p0<= p <p1:
        rho = pow(p/K1,1/Gamma1)
        eps = (1.0+alpha1)*rho + K1/(Gamma1-1.0)*pow(rho,Gamma1)
    elif p1<= p <p2:
        rho = pow(p/K2,1/Gamma2)
        eps = (1.0+alpha2)*rho + K2/(Gamma2-1.0)*pow(rho,Gamma2)
    else:
        rho = pow(p/K3,1/Gamma3)
        eps = (1.0+alpha3)*rho + K3/(Gamma3-1.0)*pow(rho,Gamma3)
    return eps*c**2


################################################################################
## THE TOV INTEGRATOR
################################################################################
t_p_2 = (G / c**4) * 1.e10 

def TOV1(logrho_c, theta):
    logp, Gamma1, Gamma2, Gamma3 = theta
    rho_samples = np.logspace(8, 16.5, 500) 
    p_eos = []
    e_eos = []
    for r_val in rho_samples:
        p_val, e_val = p_eps_of_rho(r_val, logp, Gamma1, Gamma2, Gamma3)
        p_eos.append(p_val * t_p_2) 
        e_eos.append(e_val * t_p_2)
    p_eos = np.array(p_eos, dtype=np.float64)
    e_eos = np.array(e_eos, dtype=np.float64)
    rho_c = 10**logrho_c
    pc_cgs, ec_cgs = p_eps_of_rho(rho_c, logp, Gamma1, Gamma2, Gamma3)
    pc_geometric = pc_cgs * t_p_2
    M, R, L = tov_tide.tov_tide(e_eos, p_eos, pc_geometric)
    return M, R, L


################################################################################
## MODEL DEFINITION
################################################################################
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
    def forward(self, x):
        out = self.act(self.fc(x))
        return x + out

class PhysicsEmulator(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=512):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.block1 = ResNetBlock(hidden_dim)
        self.block2 = ResNetBlock(hidden_dim)
        self.block3 = ResNetBlock(hidden_dim)
        self.block4 = ResNetBlock(hidden_dim)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
    def forward(self, x):
        x_hidden = self.input_layer(x)
        x_hidden = self.block1(x_hidden)
        x_hidden = self.block2(x_hidden)
        x_hidden = self.block3(x_hidden)
        x_hidden = self.block4(x_hidden)
        return self.final_layer(x_hidden)
    

################################################################################
## CONFIGURATION AND HELPERS 
################################################################################
RADIUS_SCALE = 25.0
MASS_SCALE = 3.5
NUM_SAMPLES = 400000
BATCH_SIZE = 256
MODEL_NO = 1

save_dir = f"{NUM_SAMPLES}files"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
model = PhysicsEmulator().to(DEVICE)
model.load_state_dict(torch.load("Best_EOS_Model.pth"))
model.eval()


################################################################################
## SAMPLE THE REQURIED NUMBER OF EOS-CURVES FOR BENCHMARKING
################################################################################
Gamma1_values = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8]
Gamma2_values = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8]
Gamma3_values = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8]

logp_values = [33.6, 33.8, 34.0, 34.2, 34.4, 34.6, 34.8]
logrho_c_array = np.linspace(14.5, 15.4, 100)

param_combinations = list(itertools.product(Gamma1_values, Gamma2_values, Gamma3_values, logp_values))
sampled_combinations = random.sample(param_combinations, 1000)

X_mean = torch.load("X_eos_mean.pt").to(DEVICE)
X_std = torch.load("X_eos_std.pt").to(DEVICE)


################################################################################
## DEFINE BENCHMARKING FUNCTION
################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def benchmark(model, sampled_combinations, logrho_c_array,
              X_mean, X_std, DEVICE,
              MASS_SCALE, RADIUS_SCALE, run_label="Benchmark"):

    tov_total = 0.0
    model_total = 0.0
    
    # Inner progress bar for individual curves
    pbar = tqdm(sampled_combinations, desc=f"  ↳ {run_label}", leave=False)

    for (Gamma1, Gamma2, Gamma3, logp) in pbar:
        # TOV timing
        start_tov = time.perf_counter()
        for logrho_c in logrho_c_array:
            TOV1(logrho_c, [logp, Gamma1, Gamma2, Gamma3])
        tov_time = time.perf_counter() - start_tov
        tov_total += tov_time

        # Emulator timing
        start_model = time.perf_counter()
        N = len(logrho_c_array)
        x = torch.tensor(
            np.column_stack([
                logrho_c_array,
                np.full(N, logp),
                np.full(N, Gamma1),
                np.full(N, Gamma2),
                np.full(N, Gamma3)
            ]),
            dtype=torch.float32
        ).to(DEVICE)

        x_norm = (x - X_mean) / X_std
        with torch.no_grad():
            pred = model(x_norm)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        
        model_time = time.perf_counter() - start_model
        model_total += model_time
        
        # Live stats on the progress bar
        if model_time > 0:
            pbar.set_postfix({"inst_speedup": f"{tov_time/model_time:.1f}x"})

    n_curves = len(sampled_combinations)
    return {
        "tov_total": tov_total,
        "model_total": model_total,
        "tov_avg": tov_total / n_curves,
        "model_avg": model_total / n_curves,
        "speedup": (tov_total / n_curves) / (model_total / n_curves)
    }


################################################################################
## RUN BENCHMARK WITH WARM UP AND MEDIAN STATISTICS
################################################################################
N_RUNS = 10
all_tov_avgs = []
all_model_avgs = []
all_speedups = []

print("\nStarting Benchmark Suite...")

# 1. WARM-UP PHASE
print("Initializing (Warm-up Run)...")
_ = benchmark(
    model, sampled_combinations, logrho_c_array,
    X_mean, X_std, DEVICE, MASS_SCALE, RADIUS_SCALE, run_label="Warm-up"
)

# 2. TIMED TRIALS
# Main progress bar for the Trial number
main_pbar = tqdm(range(N_RUNS), desc="Overall Progress")

for i in main_pbar:
    res = benchmark(
        model,
        sampled_combinations,
        logrho_c_array,
        X_mean,
        X_std,
        DEVICE,
        MASS_SCALE,
        RADIUS_SCALE,
        run_label=f"Trial {i+1}/{N_RUNS}"
    )
    
    all_tov_avgs.append(res["tov_avg"])
    all_model_avgs.append(res["model_avg"])
    all_speedups.append(res["speedup"])
    
    main_pbar.set_postfix({"median_speedup": f"{np.median(all_speedups):.1f}x"})

# 3. COMPUTE MEDIAN STATISTICS
tov_median = np.median(all_tov_avgs)
model_median = np.median(all_model_avgs)
speedup_median = np.median(all_speedups)
speedup_std = np.std(all_speedups)

print("\n" + "="*20 + " Final Benchmark Results " + "="*20)
print(f"Number of EOS tested: {len(sampled_combinations)}")
print(f"Samples per curve:     {len(logrho_c_array)}")
print(f"Number of trials:      {N_RUNS} (after 1 warm-up)")
print("-" * 65)
print(f"Median TOV curve time:   {tov_median:.4f} s")
print(f"Median Model curve time: {model_median:.6f} s")
print(f"\n---> Speedup Factor: {speedup_median:.1f}x ± {speedup_std:.1f}x faster")
print("=" * 65)

################################################################################
################################################################################