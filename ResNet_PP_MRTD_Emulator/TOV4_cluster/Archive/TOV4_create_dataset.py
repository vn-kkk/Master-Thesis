#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


import numpy as np

from tqdm import trange
from numba import jit

import tov_4   # your f2py module


msun=147660                 # Solar mass in cm given by the formula G*M_sun/c^2

c=2.9979e10                 # speed of light in cm/s (CGS)
G=6.67408e-8                # gravitational constant in cm^3/gm/s^2 (CGS)

dkm = 1.3234e-06            # conversion of MeV/fm^3 to km^-2
dcgs = 1.78e12              # factor to convert from MeV/fm^3 to gm/cm^3
conv = 8.2601e-40           # dyn/cm^2 to km^-2
cgs1=1.7827e+12             # MeV/fm3 to gms/cm3
cgs2=1.6022e+33             # MeV/fm3 to dyne/cm2


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


# Compute the offsets specific internal energy (epsL_i) and alphaL_i at the 
# boundaries
# The general form used: ε(ρ)=(1+α)ρ+K/(Γ−1)ρ^Γ. 
# Solving for alpha ensures matching across boundaries.
# Energy density needs an additive offset to enforce continuity.

epsL_4 = 0.0
alphaL_4 = 0.0
epsL_3 = (1+alphaL_4)*rhoL_3 + KL_4/(GammaL_4 - 1.)*pow(rhoL_3, GammaL_4)
alphaL_3 = epsL_3/rhoL_3 - 1.0 - KL_3/(GammaL_3 - 1.)*pow(rhoL_3, GammaL_3 -1.0)
epsL_2 = (1+alphaL_3)*rhoL_2 + KL_3/(GammaL_3 - 1.)*pow(rhoL_2, GammaL_3)
alphaL_2 = epsL_2/rhoL_2 - 1.0 - KL_2/(GammaL_2 - 1.)*pow(rhoL_2, GammaL_2 -1.0)
epsL_1 = (1+alphaL_2)*rhoL_1 + KL_2/(GammaL_2 - 1.)*pow(rhoL_1, GammaL_2)
alphaL_1 = epsL_1/rhoL_1 - 1.0 - KL_1/(GammaL_1 - 1.)*pow(rhoL_1, GammaL_1 -1.0)

# Density thresholds for high-density polytropes
rho1 = pow(10,14.7) # Break Density 1
rho2 = pow(10,15.0) # Break Density 2

# GR conversion prefactors to go from cgs pressure/energy-density units into 
# geometric units (where G=c=1)
t_p=G/c**4
t_rho=G/c**2


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


# --- Constants ---
G = 6.6743e-8
c = 2.99792458e10
# Note: Ensure geom_conv perfectly scales cgs dynes/cm^2 to 1/km^2
geom_conv = (G / c**4) * 1e10  
M_sun_km = 1.476625  

from scipy.interpolate import PchipInterpolator
def build_eos(theta, N_rho=50000, N_smooth=300000):
    """
    Improved EoS table builder using Monotonic Cubic Spline Interpolation.
    """
    # 1. High-resolution sampling of the raw EoS
    rho_samples = np.logspace(8, 16.7, N_rho)

    # Vectorized call to your physics function
    vec_func = np.vectorize(lambda r: p_eps_of_rho(r, *theta), otypes=[float, float])
    p_raw, e_raw = vec_func(rho_samples)

    # Apply conversions
    p_raw *= geom_conv
    e_raw *= geom_conv

    # 2. Log-Log Monotonic Cubic Spline (PCHIP)
    # PCHIP is superior to np.interp because it ensures a smooth first derivative 
    # (sound speed) without introducing unphysical oscillations.
    logp = np.log(p_raw)
    loge = np.log(e_raw)
    
    # Create a smooth, high-resolution pressure grid
    logp_smooth = np.linspace(logp[0], logp[-1], N_smooth)
    
    # Fit the spline
    interp_func = PchipInterpolator(logp, loge)
    loge_smooth = interp_func(logp_smooth)
    
    p_smooth = np.exp(logp_smooth)
    e_smooth = np.exp(loge_smooth)

    # 3. Thermodynamic Consistency Check
    # Ensure epsilon > p and monotonicity
    # The Fortran solver requires e(p) to be strictly increasing.
    mask = np.diff(p_smooth, prepend=p_smooth[0] - 1e-30) > 0
    
    # Final check: e must be greater than p (Relativistic causality)
    causality_mask = e_smooth[mask] > p_smooth[mask]
    
    return np.ascontiguousarray(p_smooth[mask][causality_mask], dtype=np.float64), \
           np.ascontiguousarray(e_smooth[mask][causality_mask], dtype=np.float64)


def TOV4(logrho_c_array, theta, p_array=None, e_array=None):
    if p_array is None or e_array is None:
        p_array, e_array = build_eos(theta)

    rho_c = 10**np.array(logrho_c_array)
    pc_geom = np.array([p_eps_of_rho(r, *theta)[0] * geom_conv for r in rho_c])

    # Clip central pressures to lie securely within the EOS table limits
    pmin, pmax = p_array.min() * 1.001, p_array.max() * 0.999
    pc_geom = np.clip(pc_geom, pmin, pmax)
    
    # Cast pc_geom to contiguous float64 for Fortran
    pc_geom = np.ascontiguousarray(pc_geom, dtype=np.float64)

    # Call refactored Fortran module
    m, r, td = tov_4.tov_module.tov_mrl_vector(e_array, p_array, pc_geom)

    valid = (~np.isnan(m)) & (m > 0)
    if not np.any(valid):
        print("No valid stars returned. Check EOS bounds.")
        return np.array([]), np.array([]), np.array([])

    m, r, td = m[valid], r[valid], td[valid]
    
    # Unstable branch filtering is intact
    dm = np.diff(m)
    unstable_idx = np.where(dm <= 0)[0]
    idx_max = unstable_idx[0] + 1 if len(unstable_idx) > 0 else len(m)

    return m[:idx_max], r[:idx_max], td[:idx_max]


NUM_SAMPLES = 100000 # Number of EOS samples
# Directory to save/load dataset and models 
# If not defined, model will save in current directory.
save_dir_dataset = f"Datasets/{NUM_SAMPLES}samples"


# ==========================================================
# 1. Define ranges for input parameters
# ==========================================================
# Generate random EoS parameters
Gamma_samples = np.random.uniform(low=1.4, high=5.0, size=(NUM_SAMPLES, 3))
logp_samples = np.random.uniform(33.5, 34.8, size=(NUM_SAMPLES, 1))
logrho_c_samples = np.random.uniform(14.5, 15.4, size=(NUM_SAMPLES, 1))

# Combine for easier indexing: [logrho_c, logp, G1, G2, G3]
raw_inputs = np.hstack([logrho_c_samples, logp_samples, Gamma_samples])

# ==========================================================
# 2. Solving TOV Loop
# ==========================================================
dataset_list = []

for i in trange(NUM_SAMPLES, desc="Generating Dataset"):
    logrho_c = raw_inputs[i, 0]
    logp = raw_inputs[i, 1]
    g1, g2, g3 = raw_inputs[i, 2], raw_inputs[i, 3], raw_inputs[i, 4]
    
    try:
        # Build EoS table
        p_table, e_table = build_eos([logp, g1, g2, g3])
        
        # TOV4 returns ARRAYS (m, r, td) because it filters for stability
        m_arr, r_arr, td_arr = TOV4([logrho_c], [logp, g1, g2, g3], p_table, e_table)
        
        # Check if we got a valid star back (m_arr won't be empty)
        if len(m_arr) > 0:
            # Extract scalars from the returned arrays
            M, R, L = float(m_arr[0]), float(r_arr[0]), float(td_arr[0])
            
            # Physical validation
            if (0.15 < M < 3.5) and (6.0 < R < 25.0) and (L > 0):
                dataset_list.append([logrho_c, logp, g1, g2, g3, M, R, L])
        
        # Debug first few
        if i < 5 and len(m_arr) > 0:
            print(f"Sample {i}: M={m_arr[0]:.2f}, R={r_arr[0]:.2f}")
        elif i < 5:
            print(f"Sample {i}: No stable star found at this rho_c.")

    except Exception as e:
        # This catches any indexing or math errors during the process
        continue

# ==========================================================
# 4. Finalize and Save
# ==========================================================
final_dataset = np.array(dataset_list)

if len(final_dataset) > 0:
    print(f"Dataset Generation Complete!")
    print(f"Kept {len(final_dataset)} / {NUM_SAMPLES} samples ({len(final_dataset)/NUM_SAMPLES:.1%})")

    os.makedirs(save_dir_dataset, exist_ok=True)
    filename = os.path.join(save_dir_dataset, f"EOS_dataset_{NUM_SAMPLES}files.npy")
    np.save(filename, final_dataset)
    print(f"Dataset saved to: {filename}")
else:
    print("CRITICAL: No samples were kept. Check if logrho_c is consistently inside the stable branch.")
