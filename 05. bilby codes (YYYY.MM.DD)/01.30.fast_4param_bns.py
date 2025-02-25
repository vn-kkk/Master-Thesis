#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import bilby


############################## SETTING UP SIGNAL ###############################


# Specify the logger and the output directory.
label = "3.3 fast_4param_bns"
outdir = "outdir_"+label
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.


############################# INJECTION PARAMETERS #############################


injection_parameters = dict(
# Mass parameters
    mass_1=1.5, # Masses of Neutron stars, in solar masses.
    mass_2=1.3,

# Spin parameters
    chi_1=0.02, # Dimensionless spin magnitudes.
    chi_2=0.02,
    
# Distance parameter
    luminosity_distance=50.0,  # Distance to the binary system, in Mpc.

# Angular Orientation parameters
    theta_jn=0.4, # Inclination angle, i.e. the angle between the total
                  # angular momentum vector and the line of sight, in radians

    psi=2.659,  # The polarization angle, describing the orientation of the 
                # binary's orbital plane relative to the observer, in radians.
             
    phase=1.3,  # The orbital phase of the binary at coalescence, in radians.

# Time parameter
    geocent_time=1126259642.413, # The time of the signal at the Earth's center, 
                                 # in GPS seconds.
                                 # This corresponds to the moment of coalescence

# Sky Location parameters
    ra=1.375,   # The right ascension (RA) of the binary system on 
                # the celestial sphere, in radians.

    dec=-1.2108,# The declination (Dec) of the binary system on 
                # the celestial sphere, in radians.

# Tidal Deformability parameters
    lambda_1=545,
    lambda_2=1346,
)


################################ WAVEFORM MODEL ################################


# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 32 # seconds
sampling_frequency = 2048.0 # in Hz
minimum_frequency = 40 # below this the detector sensitivity is poor
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)


################################### DETECTOR ###################################


# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for interferometer in ifos:
    interferometer.minimum_frequency = 40
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)


############################## BAYESIAN INFERENCE ##############################


# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The below list does *not* include *mass_1*, *mass_2*, *luminosity_distance*, 
# and *theta_jn* which means those are the parameters that will be included in
# the sampler. If we do nothing, then the default priors get used.


#################################### PRIORS ####################################


priors = bilby.gw.prior.BNSPriorDict()
# We are taking only mass_1, mass_2, chirp_mass, lambda_1, lambda_2,
# luminosity_distance, theta_jn from the BNSPriorDict()

for key in [
    "chi_1",
    "chi_2",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key] # setting their values to be equal 
                                            # to the injection parameters

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


# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)


################################### SAMPLER ####################################


# Run sampler.  In this case we're going to use the `dynesty` sampler
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

# Make a corner plot.
result.plot_corner()
