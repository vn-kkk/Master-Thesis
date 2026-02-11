################################################################################
################################################################################
# IMPORT ALL REQUIRED MODULES
################################################################################
################################################################################
import os

import numpy as np
 
from tqdm import trange
from numba import jit

import tov_tide


################################################################################
################################################################################
# GLOBAL CONSTANTS AND UNIT CONVERSION FACTORS
################################################################################
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
################################################################################
# 4 PIECEWISE POLYTROPIC LOW-DENSITY (CRUST) PARAMETERS 
################################################################################
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

# Pressures at the boundaries of the low denisty piecewise polytropes
pL_1 = KL_1*rhoL_1**GammaL_1
pL_2 = KL_2*rhoL_2**GammaL_2
pL_3 = KL_3*rhoL_3**GammaL_3
pL_4 = 0.0

# The exact numbers are taken from a particular crust model/table.


################################################################################
################################################################################
# LOW DENSITY ENERGY DENISTY, α, AND BREAK DENSITIES
################################################################################
################################################################################
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


################################################################################
################################################################################
# FORWARD EOS: FROM (logp, Gamma1, Gamma2, Gamma3) TO p(ρ) AND ε(ρ)
# To calculate pressure and energy density for polytropes
# based on the central density region of intrest
################################################################################
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
################################################################################
# INVERSE EOS: FROM (logp, Gamma1, Gamma2, Gamma3) TO ε(p)
# To calculate Energy denisty for every central pressure value of intrest 
# while solving a polytrope
################################################################################
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
################################################################################
# THE TOV INTEGRATOR
################################################################################
################################################################################
def TOV(logrho_c, theta, compute_tidal=True):
    logp, Gamma1, Gamma2, Gamma3 = theta
    dr = 100.0              # radial step in meters

    rho_c = 10**logrho_c    # Central density in cgs units (g/cm^3)
    r = 0.1                 # Initial Radius (Non-zero to avoid singularity)
    m = 0.0                 # Initial Mass

    p, e = p_eps_of_rho(rho_c, logp, Gamma1, Gamma2, Gamma3)
    p *= t_p                # pressure in geometric units
    e *= t_p                # energy density in geometric units

    # Store profiles to calculate TD
    p_prof = []
    e_prof = []
    r_prof = []
    m_prof = []

    while p > 0:    # As long as pressure is greater than 0
        p_prof.append(p)
        e_prof.append(e)
        r_prof.append(r)
        m_prof.append(m)

        dp = -(e + p) * (m + 4*np.pi*r**3*p) / (r*(r - 2*m))
        p += dp * dr    # Update pressure at each radial step
        if p <= 0:      # Break when pressure reaches 0
            break

        m += 4*np.pi*r**2 * e * dr      # Update mass
        r += dr                         # Update radius
        e = eps(p/t_p, logp, Gamma1, Gamma2, Gamma3) * t_p  # Update energy density

    # --- Final mass and radius ---
    M = m / msun
    R = r / 1e5

    if not compute_tidal:
        return M, R

    # prepare inputs for Fortran tov_tide.tov_tide
    p_prof = np.array(p_prof, dtype=np.float64)
    e_prof = np.array(e_prof, dtype=np.float64)
    # Fortran expects central pressure at index N
    p_prof = p_prof[::-1]
    e_prof = e_prof[::-1]
    pc = p_prof[-1]
    # N = len(p_prof)

    # --- Calculate tidal deformability ---
    M_tide, R_tide, Lambda = tov_tide.tov_tide(
        e_prof,
        p_prof,
        pc
    )
    
    return M, R, Lambda     # Returns true TD (not log TD)


################################################################################
################################################################################
# CREATE DATASET FOR TRAINING THE MODEL
################################################################################
################################################################################
NUM_SAMPLES = 200    # Number of EOS samples
# Directory to save/load dataset and models 
# If not defined, model will save in current directory.
save_dir = f"{NUM_SAMPLES}samples"

# ==========================================================
# Define ranges for input parameters
# ==========================================================
EOS_params = np.random.uniform( low=[1.4, 1.4, 1.4], 
                                high=[5., 5., 5.], 
                                size=(NUM_SAMPLES, 3)
                                )
logp_samples = np.random.uniform(33.5, 34.8, size=(NUM_SAMPLES, 1))


logrho_c_samples = np.random.uniform(14.5, 15.4, size=(NUM_SAMPLES, 1))

MRL_data = []

for i in trange(NUM_SAMPLES, desc="Solving TOV"):
    logrho_c = logrho_c_samples[i, 0]
    logp = logp_samples[i, 0]
    params = EOS_params[i]

    M, R, Lambda = TOV(
        logrho_c,
        [logp, params[0], params[1], params[2]],
        compute_tidal = True
        )
    MRL_data.append([M, R, Lambda])

MRL_data = np.array(MRL_data)

# ==========================================================
# Apply mask to filter out unphysical outputs produced by extremely stiff EOSs
# ==========================================================
M = MRL_data[:, 0]
R = MRL_data[:, 1]
Lambda = MRL_data[:, 2]
mask = (
    np.isfinite(M) &
    np.isfinite(R) &
    (M > 0.15) & (M < 3.5) &
    (R > 6.0) & (R < 25.0) &
    (Lambda > 0) & (Lambda < 1e6)
)

# Stack the cleaned dataset
EOS_data = np.hstack([
    logrho_c_samples[mask],
    logp_samples[mask],
    EOS_params[mask],
    MRL_data[mask]
])
print(f"Kept {EOS_data.shape[0]} / {NUM_SAMPLES} samples")

# Save cleaned dataset
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, f"EOS_dataset_{NUM_SAMPLES}samples.npy"), EOS_data)
print("Datasets created and saved!")


# When creating a dataset, check the error file produced by the cluster to see the progress
