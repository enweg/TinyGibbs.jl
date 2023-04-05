# TinyGibbs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://enweg.github.io/TinyGibbs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://enweg.github.io/TinyGibbs.jl/dev/)
[![Build Status](https://github.com/enweg/TinyGibbs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/enweg/TinyGibbs.jl/actions/workflows/CI.yml?query=branch%3Amain)

*TinyGibbs* is a small Gibbs sampler that makes use of the *AbstractMCMC* interface. It therefore allows for efficient Gibbs sampling including parallel sampling of multiple chains. Additionally, *TinyGibbs* can collect samples in two ways: (1) as a dictionary of tensors where each tensor or (2) as a *MCMCChains.Chains* type. Therefore, all the funcionality of *MCMCChains* can be exploited with *TinyGibbs*. 

> *TinyGibbs* goal is to be intuitive and as close as possible to research papers. That is, the goal is to have a syntax that is close to the notation used for Gibbs sampling procedures in research papers

## How does it work? 

```jl
using TinyGibbs
using StableRNGs
using Random, Distributions
using MCMCChains, AbstractMCMC
using LinearAlgebra
```

To achieve its goal of being as close as possible to research paper notation, *TinyGibbs* introduced the `@tiny_gibbs` macro. This macro allows one to abstract away all the computational elements and to strictly focus on the Gibbs step logic - that is, on the way in which each parameter is drawn given the other parameters. 

As an example, consider the Multivariate Normal Distribution 

$$
\begin{bmatrix}X \\ Y \end{bmatrix} \sim N(\mu, \Sigma)
$$

where 

$$
\mu = \begin{bmatrix}\mu_X \\ \mu_Y\end{bmatrix}
$$

and 

$$
\Sigma = \begin{bmatrix}\Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY}\end{bmatrix}
$$

we then have the following rules: 

💡 **Rules for multivariate normal distribution**

$$
X \sim N(\mu_X, \Sigma_{XX})
$$

and 

$$ 
Y|X \sim N(\mu_Y + \Sigma_{YX}\Sigma_{XX}^{-1}(X-\mu_x),\quad \Sigma_{YY}-\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY})
$$

---

We can therefore create the following Gibbs sampling procedure

```jl
@tiny_gibbs function gibbs_normal(mu, Σ)
    # Drawing y: here a vector of all elements except the first
    my = mu[2:end] + 1/Σ[1, 1]*Σ[2:end, 1]*(x - mu[1])
    Σy = Σ[2:end, 2:end] - 1/Σ[1, 1]*Σ[2:end, 1]*Σ[1, 2:end]'
    y ~ MultivariateNormal(my, Hermitian(Σy))

    # drawing the first element conditional on the others
    mx = mu[1] + Σ[1, 2:end]'*inv(Σ[2:end, 2:end])*(y - mu[2:end])
    Σx = Σ[1, 1] - Σ[1, 2:end]'*inv(Σ[2:end, 2:end])*Σ[2:end, 1]
    x ~ Normal(mx, sqrt(Σx))
end
```

This will create a function `gibbs_normal` in our environment. This function takes as the first argument a dictionary of initial values. Each variable in the Gibbs sampling procedure that is on the LHS of a `~` must be a key in the dictionary and must therefore have an initial value. As the remaining arguments, `gibbs_normal` will take the arguments that were given in the macro - hence `mu` and `Σ`. 

```jl
# Use a stable RNG for replicability reasons
rng = StableRNG(123)
# Create some parameters
mu = rand(rng, MultivariateNormal(30*randn(rng, 3), I))
Σ = rand(rng, Wishart(4, diagm(ones(3))))
# Define initial values 
initial_values = Dict(:x => mu[1], :y => mu[2:end])
# Create a sampler
sampler = gibbs_normal(initial_values, mu, Σ)
```

After creating a sampler, we are now ready to sample. *TinyGibbs* overwrites the *AbstractMCMC.sample* methods such that there is one argument less. If the user absolutely wishes to use the *AbstractMCMC.sample* methods though, they can still do so, by using *TinyGibbsModel* as the model. 

Sampling can either be done for a single chain, or for multiple chains. In the latter case, sampling of the multiple chains can also make use of parallelization. 

```jl
# Sampling a single chain of 1000 draws and saving it as a MCMCChains.Chains type
chain_single = sample(rng, sampler, 1_000; chain_type=MCMCChains.Chains)
# Same as above, but this time saving draws as a dictionary of tensors
# The last dimensions follow the following rules
# 1. The last dimension of each tensor refers to the chain
# 2. The second to last dimension refers to the draws
# 3. The remaining dimensions are the dimensions of the sampled object, i.e. two dimensional for covariance matrices
chain_single_dict = sample(rng, sampler, 1_000; chain_type=Dict)
```

To make use of parallel sampling, we can use any of *AbstractMCMC*s methods. Here I will choose *MCMCThreads()*

```jl
# Sampling 4 chains each having 1000 draws in parallel 
chain_parallel = sample(rng, sampler, MCMCThreads(), 1_000, 4; chain_type=MCMCChains.Chains)
chain_parallel_dict = sample(rng, sampler, MCMCThreads(), 1_000, 4; chain_type=Dict)
```

We can then use these draws like any other Bayesian draws. For example, we can just plot the draws using *MCMCChains* interface

```jl
using StatsPlots
plot(chain_parallel)
```

We can also compare the Gibbs sampled distribution for `x` with the theoretical marginal distribution

```jl
histogram(chain_parallel[:x]; normalize=:pdf, legend=:none)
plot!(minimum(chain_parallel[:x]):0.01:maximum(chain_parallel[:x]), x->pdf(Normal(mu[1], sqrt(Σ[1, 1])), x); color=:red, linewidth=2)
```

## Current Shortcomings / Potential next steps

- *TinyGibbs* does not currently support the use of MH or HMC within Gibbs. A natrual next step would be to make this possible
- *TinyGibbs* does not currently support keeping track of any other quantities than those that are being sampled. This can be changed if the need ever comes up. A hack around this would also be to have a deterministic distribution. 

## TODOS

- [ ] Fill in [compats]
- [ ] Publish
- [ ] Documentation