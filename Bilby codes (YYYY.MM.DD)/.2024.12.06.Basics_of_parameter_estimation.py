#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise of unknown variance

"""
import bilby
import matplotlib.pyplot as plt
import numpy as np

# A few simple setup steps
label = "linear_regression"
outdir = "outdir_2.2 Basics_of_parameter_estimation"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which are the "true" values we plug in.
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10 #in Hz. Number of samples per second 
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency) 
# np.arrange(start, stop, step)
N = len(time) # Total number of samples
sigma = np.random.normal(1, 0.01, N) # Random values drawn from a 
# normal (Gaussian) distribution with mean= 1 and standard deviation= 0.01. 
# This array represents the standard deviation of the noise at each time point.

data = model(time, **injection_parameters) + np.random.normal(0, sigma, N) 
# np.random.normal(loc (i.e the mean or centre of the distribution), 
# scale(SD of the distribution), size (gives output shape)).
# This is the background gaussian noise

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, "o", label="data")
ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
ax.set_xlabel("time")
ax.set_ylabel("data")
ax.legend()

fig.savefig("{}/{}_data.png".format(outdir, label))

# Now lets instantiate a version of our GaussianLikelihood, 
# giving it the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma) 
# (x, y, python funtion to fit to the data, sigma) 

# From hereon, the syntax is exactly equivalent to other bilby examples
# The larger the prior range the longer the sampler takes to run
priors = dict() 
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=250, # minimum number of points for a good estimate is 500
               # 100 is better for gravitational wave problems
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()
