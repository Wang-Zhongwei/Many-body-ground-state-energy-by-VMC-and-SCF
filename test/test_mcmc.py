import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# Define the function to integrate
def integrand(x):
  return 1/(x**2 + 1)

# Define the function to sample from (the proposal distribution)
def proposal(x):
  return x + tf.random.normal(shape=x.shape, stddev=0.1)

# Define the starting point for the Markov Chain
initial_state = tf.constant(0.)

# Create the Markov Chain Monte Carlo method
mcmc = tfp.mcmc.MetropolisHastings(
    target_log_prob_fn=integrand,
    proposal_fn=proposal,
    num_leapfrog_steps=10)

# Run the Markov Chain and store the samples
samples, _ = tfp.mcmc.sample_chain(
    num_results=10000,
    num_burnin_steps=1000,
    current_state=initial_state,
    kernel=mcmc)

# Calculate the mean of the samples as an estimate of the integral
mean = tf.reduce_mean(samples)
print(mean)




