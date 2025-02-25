#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data consisting of a Gaussian with a mean and variance
"""
import bilby
import numpy as np
import matplotlib.pyplot as plt


# A few simple setup steps
label = "1 simple_gausian_likelihood"
outdir = "outdir_"+label
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# Here is minimum requirement for a Likelihood class to run with bilby. In this
# case, we setup a GaussianLikelihood, which needs to have a log_likelihood
# method. Note, in this case we will NOT make use of the `bilby`
# waveform_generator to make the signal.

# Making simulated data: in this case, we consider just a Gaussian

data = np.random.normal(3, 4, 100) # (mean, SD, size)
mean=np.mean(data)
plt.ylim(-20,20)
plt.axhline(mean,color='r', ls='--', label='Mean= %s'%np.round(mean,2))
plt.plot(data, 'co', label='data')
plt.xlabel('data points')
plt.ylabel('SD')
plt.legend()
plt.savefig("{}/{}_data.png".format(outdir, label))


# class SimpleGaussianLiklelihood is inheriting from bilby.liklelihood
class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data): # this is a constructor
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={"mu": None, "sigma": None})
        # initializes the parent class
        # mu is mean of the Gaussian distribution to be estimated 
        # and sigma is the SD
        self.data = data
        self.N = len(data)

    def log_likelihood(self): # this is a method
        # it computes the log of the likelihood function
        mu = self.parameters["mu"]
        sigma = self.parameters["sigma"]
        res = self.data - mu # residuals 
        return -0.5 * (
            np.sum((res / sigma) ** 2) + self.N * np.log(2 * np.pi * sigma**2)
        )


likelihood = SimpleGaussianLikelihood(data)
priors = dict(
    mu=bilby.core.prior.Uniform(0, 5, "mu"),
    sigma=bilby.core.prior.Uniform(0, 10, "sigma"),
)

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    outdir=outdir,
    label=label,
)
result.plot_corner()
