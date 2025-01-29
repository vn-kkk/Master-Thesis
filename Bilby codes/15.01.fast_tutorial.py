#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import bilby

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
label = "fast_tutorial"
outdir = "outdir_"+label
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
# Mass parameters
    mass_1=36.0, # Masses of black holes, in solar masses.
    mass_2=29.0,

# Spin parameters
    a_1=0.4, # Dimensionless spin magnitudes.
    a_2=0.3,

    tilt_1=0.5, # The angle between the spin vector of the black holes
    tilt_2=1.0, # and the orbital angular momentum, in radians.
                # (0 means aligned, π means anti-aligned).

    phi_12=1.7, # The azimuthal angle between the spin vectors of the two 
                # black holes, in radians.
    
    phi_jl=0.3, # The azimuthal angle between the total angular momentum and 
                # the orbital angular momentum, in radians.
    
# Distance parameter
    luminosity_distance=2000.0,  # Distance to the binary system, in Mpc.

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

#Sky Location parameters
    ra=1.375,   # The right ascension (RA) of the binary system on 
                # the celestial sphere, in radians.

    dec=-1.2108,# The declination (Dec) of the binary system on 
                # the celestial sphere, in radians.
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

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
priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key] # setting their values to be equal 
                                            # to the priors

# Perform a check that the prior does not extend to a parameter space 
# longer than the data
priors.validate_prior(duration, minimum_frequency)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=1000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Make a corner plot.
result.plot_corner()
