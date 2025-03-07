#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a binary neutron star
system taking into account tidal deformabilities.

This example estimates the masses using a uniform prior in both component masses
and also estimates the tidal deformabilities using a uniform prior in both
tidal deformabilities
"""


import bilby
import numpy as np


############################## SETTING UP SIGNAL ###############################


# Specify the output directory and the name of the simulation.
label = "3.2 bns_example"
outdir = "outdir_"+label
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.


############################# INJECTION PARAMETERS #############################


injection_parameters = dict(
# Mass parameters
    mass_1=1.5,
    mass_2=1.3,

# Spin parameters
    chi_1=0.02,
    chi_2=0.02,

# Distance parameter
    luminosity_distance=50.0,

# Angular Orientation parameters
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,

# Time parameter
    geocent_time=1126259642.413,

# Sky Location parameters
    ra=1.375,
    dec=-1.2108,

# Tidal Deformability parameters
    lambda_1=545,
    lambda_2=1346,
)


################################ WAVEFORM MODEL ################################


# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the TaylorF2 waveform, we cut the signal 
# close to the innermost stable circular orbit (isco) frequency.
duration = 32
sampling_frequency = 2048
minimum_frequency= 40.0
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)


################################### DETECTOR ###################################


# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, 
    duration=duration, 
    start_time=start_time
)
interferometers.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)


############################## BAYESIAN INFERENCE ##############################


# Load the default prior for binary neutron stars.
# We're going to sample in 
# *chirp_mass*, *symmetric_mass_ratio*, *lambda_tilde*, and *delta_lambda_tilde* 
# rather than mass_1, mass_2, lambda_1, and lambda_2.
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict


#################################### PRIORS ####################################


priors = bilby.gw.prior.BNSPriorDict()

for key in [
    "psi",
    "geocent_time",
    "ra",
    "dec",
    "chi_1",
    "chi_2",
    "theta_jn",
    "luminosity_distance",
    "phase",
]:
    priors[key] = injection_parameters[key]

# deleting these priors because we are assigning them new values
del priors["mass_ratio"], priors["lambda_1"], priors["lambda_2"]

priors["chirp_mass"] = bilby.core.prior.Gaussian(
    1.215, 0.1, name="chirp_mass", unit="$M_{\\odot}$"
)
priors["symmetric_mass_ratio"] = bilby.core.prior.Uniform(
    0.1, 0.25, name="symmetric_mass_ratio"
)
priors["lambda_tilde"] = bilby.core.prior.Uniform(0, 5000, name="lambda_tilde")

priors["delta_lambda_tilde"] = bilby.core.prior.Uniform(
    -500, 1000, name="delta_lambda_tilde"
)


# constraining these parameters to prevent out of bounds sampling.
priors["lambda_1"] = bilby.core.prior.Constraint(
    name="lambda_1", minimum=0, maximum=10000
)
priors["lambda_2"] = bilby.core.prior.Constraint(
    name="lambda_2", minimum=0, maximum=10000
)



################################## LIKELIHOOD ##################################


# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
)


################################### SAMPLER ####################################


# Run sampler.  In this case we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=100,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
)

result.plot_corner()
