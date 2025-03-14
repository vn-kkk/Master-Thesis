#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal, using the relative binning likelihood.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""
from copy import deepcopy # creates independent copies of objects

import bilby
import numpy as np
from tqdm.auto import trange # to display progress bar for loops


############################## SETTING UP SIGNAL ###############################


# Specify the output directory and the name of the simulation.
label = "4.3 relative_binning_BNS"
outdir = "outdir_"+label
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)


# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
# Mass parameters
    mass_1=1.5,
    mass_2=1.3,

# Spin parameters
    chi_1=0.2,
    chi_2=0.2,

# Distance parameter
    luminosity_distance=2000.0,

# Angular orientation parameters
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,

# Time parameter
    geocent_time=1126259642.413,

# Sky Location
    ra=1.375,
    dec=-1.2108,

# Tidal Deformability parameters
    lambda_1=545.0,
    lambda_2=1346.0,

# Serves as a reference marker for the injection parameters.
    fiducial=1, # Used while reweighting to identify the injection parameters
)


################################ WAVEFORM MODEL ################################


# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 32
sampling_frequency = 4096.0
minimum_frequency = 75.0
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal", # A phenomenological 
                         # Inspiral-Merger-Ringdown model for a BNS
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)



# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    # frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)


################################### DETECTOR ###################################


# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1) and Virgo (V1)).
# These default to their design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in ifos:
    interferometer.minimum_frequency = 75
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)


############################## BAYESIAN INFERENCE ##############################


# Load the default prior for binary neutron stars.
# We're going to sample in 
# *chirp_mass*, *mass_ratio*, *lambda_tilde*, and *delta_lambda_tilde* 
# rather than mass_1, mass_2, lambda_1, and lambda_2.
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict

# The 4 output parameters are:

# Mass ratio (q)
# Chirp mass (M)
# Luminosity distance (d_L)
# Inclination angle (theta_JN)
# lambda tilde 
# chi_eff

#################################### PRIORS ####################################


priors = bilby.gw.prior.BNSPriorDict()

for key in [
    "geocent_time",
    "chi_1",
    "chi_2",
    "ra",
    "dec",
    "psi",
    "phase",
]:
    priors[key] = injection_parameters[key]

# Perform a check that the prior does not extend to a parameter space 
# longer than the data
priors.validate_prior(duration, minimum_frequency)

# deleting these priors because we are assigning them new values
del priors["lambda_1"], priors["lambda_2"], priors["mass_ratio"], priors["chirp_mass"]

priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
    minimum=0.4, maximum=2.4, name='chirp_mass', latex_label='$\\mathcal{M}$', 
    unit=None, boundary=None
)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
    minimum=0.125, maximum=1, name='mass_ratio', latex_label='$q$', unit=None, 
    boundary=None, equal_mass=False                                      
)
priors["lambda_1"] = bilby.core.prior.Constraint(
    minimum=100.0, maximum=5000.0, name='lambda_1', latex_label='$\\Lambda_1$', 
    unit=None
)
priors["lambda_2"] = bilby.core.prior.Constraint(
    minimum=100.0, maximum=5000.0, name='lambda_2', latex_label='$\\Lambda_2$', 
    unit=None
)
priors["lambda_tilde"] = bilby.core.prior.Gaussian(
    mu=867.9931562541493, sigma=1.0, name='lambda_tilde', 
    latex_label='$\\tilde{\\Lambda}$', unit=None, boundary=None
)
priors["chi_eff"] = bilby.core.prior.Sine(minimum=0, maximum=0.5, 
    name='chi_eff', latex_label='$\\chi_{eff}$', unit=None, boundary=None
)


################################## LIKELIHOOD ##################################

# chirp_mass: 1.2150360414642816
# mass_ratio: 0.8666666666666667
# lambda_tilde':867.9931562541493
# delta_lambda_tilde':95.05130053992274

# Set up the fiducial parameters for the relative binning likelihood to be the
# injected parameters. Note that because we sample in chirp mass, mass ratio,
# lambda_tilde and delta_lambda_tilde but injected with mass_1, mass_2,
# lambda_1, and lambda_2, we need to convert the mass and tidal deformability
# parameters
fiducial_parameters = injection_parameters.copy()

m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2) # Chirp mass (M) (one of the prameters being estimated)
fiducial_parameters["mass_ratio"] = m2/m1
                   # mass ratio (q) (another estimated parameter)

l1=fiducial_parameters.pop("lambda_1")
l2=fiducial_parameters.pop("lambda_2")
fiducial_parameters["lambda_tilde"] =bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
    l1,l2,m1,m2) # Lambda tilde (another prameter being estimated)
fiducial_parameters["delta_lambda_tilde"]=bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
    l1,l2,m1,m2) # Delta lambda tilde (another prameter being estimated)

fiducial_parameters["chi_eff"] =0.02

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=True,
    fiducial_parameters=fiducial_parameters,
)


################################### SAMPLER ####################################


# Run sampler.  In this case, we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
    )


####################### REWIGHTING WITH FULL LIKELIHOOD ########################


# Alternative waveform generator WITHOUT relative binning
alt_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    # frequency_domain_source_model=lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Alternative likelihood WITHOUT relative binning
alt_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=alt_waveform_generator,
)


likelihood.distance_marginalization = True

# Reweights samples from binned likelihood using the full likelihood
weights = list()
for ii in trange(len(result.posterior)):
    parameters = dict(result.posterior.iloc[ii])
    likelihood.parameters.update(parameters)
    alt_likelihood.parameters.update(parameters)
    weights.append(
        alt_likelihood.log_likelihood_ratio() - likelihood.log_likelihood_ratio()
    )
weights = np.exp(weights)

# Compute efficiency and Bayes factor between binned and unbinned methods
print(
    f"""Reweighting efficiency is 
    {np.mean(weights)**2 / np.mean(weights**2) * 100:.2f}%"""
)
print(f"Binned vs unbinned log Bayes factor {np.log(np.mean(weights)):.2f}")

# Generate result object with the posterior for the regular likelihood using
# rejection sampling
alt_result = deepcopy(result)
keep = weights > np.random.uniform(0, max(weights), len(weights))
alt_result.posterior = result.posterior.iloc[keep]

# Make a comparison corner plot.
bilby.core.result.plot_multiple(
    [result, alt_result],
    labels=["Binned (Relative binning)", "Reweighted (Unbinned likelihood)"],
    filename=f"{outdir}/{label}_corner.png",
)
