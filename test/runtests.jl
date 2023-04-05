using TinyGibbs
using StableRNGs
using Random
using Distributions
using Test
using LinearAlgebra
using MCMCChains
using Statistics

@testset "TinyGibbs.jl" begin
    """
    A test where the first distribution does not depend on any other variables.
    Test also includes non-drawing intermediate variables.
    """
    @tiny_gibbs function m()
        a ~ Normal(0, 1)
        z = sin(a)
        b ~ Normal(z, z^2)
    end
    function m_test(rng, state)
        a = rand(rng, Normal(0, 1))
        z = sin(a)
        b = rand(rng, Normal(z, z^2))
        return Dict(:a => a, :b => b)
    end
    state = Dict(:a => 1.0, :b => 1.0)
    sampler = m(state)
    macro_state = sampler.draw(StableRNG(123), state)
    test_state = m_test(StableRNG(123), state)
    @test test_state == macro_state

    """
    A test where all variables depend on other variables.
    """
    @tiny_gibbs function m2()
        a ~ Normal(b, z^2)
        b ~ Normal(a / 2, 1)
        z ~ Gamma(a^2, b^2)
    end
    function m2_test(rng, state)
        new_state = deepcopy(state)
        new_state[:a] = rand(rng, Normal(new_state[:b], new_state[:z]^2))
        new_state[:b] = rand(rng, Normal(new_state[:a] / 2, 1))
        new_state[:z] = rand(rng, Gamma(new_state[:a]^2, new_state[:b]^2))
        return new_state
    end
    state = Dict(:a => 1.0, :b => 0.5, :z => 10.0)
    sampler = m2(state)
    macro_state = sampler.draw(StableRNG(123), state)
    test_state = m2_test(StableRNG(123), state)
    @test test_state == macro_state

    """ 
    A test where the model also depends on external data
    """
    @tiny_gibbs function m3(x, y)
        a ~ Normal(x + b, y^2)
        b ~ Gamma(abs(a) / 100, abs(a) / 200)
    end
    function m3_test(rng, state, x, y)
        new_state = deepcopy(state)
        new_state[:a] = rand(rng, Normal(x + new_state[:b], y^2))
        new_state[:b] = rand(rng, Gamma(abs(new_state[:a]) / 100, abs(new_state[:a] / 200)))
        return new_state
    end
    initial_values = Dict(:a => 1.0, :b => 10.0)
    x = -1.0
    y = 1.4
    sampler = m3(initial_values, x, y)
    macro_state = sampler.draw(StableRNG(123), initial_values, sampler.data...)
    test_state = m3_test(StableRNG(123), initial_values, x, y)
    @test test_state == macro_state
    # also testing sampling function
    rng = StableRNG(123)
    macro_chain = sample(rng, sampler, 10)
    rng = StableRNG(123)
    test_chain = []
    state = initial_values 
    for i in 1:10
        state = m3_test(rng, state, x, y)
        push!(test_chain, state)
    end
    @test all(test_chain .== macro_chain)


    """ 
    Sampling in a Gibbs way from a multivariate normal and checking if marginal
    is okay.
    Test is motivated by: 
    https://github.com/stan-dev/stan/wiki/Testing:-Samplers
    """
    @tiny_gibbs function m4(mu, Σ)
        # Drawing y: here a vector of all elements except the first
        my = mu[2:end] + 1/Σ[1, 1]*Σ[2:end, 1]*(x - mu[1])
        Σy = Σ[2:end, 2:end] - 1/Σ[1, 1]*Σ[2:end, 1]*Σ[1, 2:end]'
        y ~ MultivariateNormal(my, Hermitian(Σy))

        # drawing the first element conditional on the others
        mx = mu[1] + Σ[1, 2:end]'*inv(Σ[2:end, 2:end])*(y - mu[2:end])
        Σx = Σ[1, 1] - Σ[1, 2:end]'*inv(Σ[2:end, 2:end])*Σ[2:end, 1]
        x ~ Normal(mx, sqrt(Σx))
    end
    rng = StableRNG(123)
    mu = rand(rng, MultivariateNormal(30*randn(rng, 3), I))
    Σ = rand(rng, Wishart(4, diagm(ones(3))))
    initial_values = Dict(:x => mu[1], :y => mu[2:end])
    sampler = m4(initial_values, mu, Σ)
    chain = sample(rng, sampler, 1_000; chain_type=MCMCChains.Chains)

    mcse_mean = MCMCChains.mcse(reshape(chain[:x].data, (size(chain[:x].data)..., 1)); kind=Statistics.mean)[1]
    mcse_std = MCMCChains.mcse(reshape(chain[:x].data, (size(chain[:x].data)..., 1)); kind=Statistics.std)[1]

    abs_diff_mean = abs(mean(chain[:x]) - mu[1])
    delta_mean = abs_diff_mean/mcse_mean
    p_value_mean = cdf(Normal(0, 1), -delta_mean)*2
    @info "Mean --> abs_diff=$abs_diff_mean, p_value=$p_value_mean"
    @test p_value_mean > 0.05

    abs_diff_std = abs(std(chain[:x]) - sqrt(Σ[1,1]))
    delta_std = abs_diff_std/mcse_std
    p_value_std =cdf(Normal(0, 1), -delta_std)*2
    @info "STD --> abs_diff=$abs_diff_std, p_value=$p_value_std"
    @test p_value_std > 0.05
end
