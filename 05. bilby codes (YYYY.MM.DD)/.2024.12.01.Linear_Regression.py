#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise with known variance.

"""
import bilby
import matplotlib.pyplot as plt
import numpy as np

# A few simple setup steps to check if an output directory exists
label = "2.1 linear_regression_unknown_noise"
outdir = "outdir_"+label
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters with which we make simulated data.
injection_parameters = dict(m=0.5, c=0.2) 

# For this example, we'll inject standard Gaussian noise with a varaince of 1
sigma = 1

# These lines of code generate the fake data. 
# Note the ** just unpacks the contents of the dictonary 
# when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time) # Number of samples
data = model(time, **injection_parameters) + np.random.normal(0, sigma, N) 
# now 'data' contains the signal mixed with some gausian noise.

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, "og", label="data") # the data points with noise
ax.plot(time, model(time, **injection_parameters), "--r", label="signal") 
# Line of Best fit which is the signal
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.legend()
fig.suptitle("Signal fitted to the data")
fig.savefig("{}/{}_data.png".format(outdir, label))

injection_parameters.update(dict(sigma=1))

# Now lets instantiate the built-in GaussianLikelihood, giving it
# the time, data and signal model. Note that, because we do not give it the
# parameter, sigma is unknown and marginalised over during the sampling
likelihood = bilby.core.likelihood.GaussianLikelihood(time, data, model)

priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")
priors["sigma"] = bilby.core.prior.Uniform(0, 10, "sigma")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty", # Nested sampler
    npoints=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()
