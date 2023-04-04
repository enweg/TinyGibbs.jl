module TinyGibbs
using AbstractMCMC
using Random
using MacroTools
using Distributions
import StatsBase

struct TinyGibbsSampler <: AbstractMCMC.AbstractSampler
    initial_values::Dict
    draw::Function
    data::AbstractArray
end
struct TinyGibbsModel <: AbstractMCMC.AbstractModel end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::TinyGibbsModel,
    sampler::TinyGibbsSampler;
    kwargs...
)
    v = sampler.draw(rng, sampler.initial_values, sampler.data...)
    return v, v
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::TinyGibbsModel,
    sampler::TinyGibbsSampler,
    state;
    kwargs...
)
    v = sampler.draw(rng, state, sampler.data...)
    return v, v
end

function StatsBase.sample(
    rng::Random.AbstractRNG, 
    sampler::TinyGibbsSampler, 
    parallel::AbstractMCMC.AbstractMCMCEnsemble, 
    N::Integer, 
    nchains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(rng, TinyGibbsModel(), sampler, parallel, N, nchains; kwargs...)
end

function StatsBase.sample(
    rng::Random.AbstractRNG, 
    sampler::TinyGibbsSampler, 
    N_or_isdone; 
    kwargs...
)
    return AbstractMCMC.sample(rng, TinyGibbsModel(), sampler, N_or_isdone; kwargs...)
end


macro tiny_gibbs(expr)
    return esc(tiny_gibbs(expr))
end
function tiny_gibbs(expr)
    def = MacroTools.splitdef(expr)
    vars = _collect_vars(def[:body])
    data = def[:args]
    def[:body] = _replace_sampling(def[:body])
    def[:body] = _replace_variables(def[:body], vars)
    pushfirst!(def[:body].args, :(new_state = deepcopy(state)))
    push!(def[:body].args, :(return new_state))
    def[:args] = vcat([:(rng::Random.AbstractRNG), :(state::Dict)], def[:args])
    f = MacroTools.combinedef(def)
    # return MacroTools.combinedef(def)
    def_construct = Dict(
        :name => def[:name],
        :args => vcat(:(initial_values::Dict), copy(data)),
        :kwargs => Any[], 
        :body => quote
            return TinyGibbsSampler(
                initial_values, 
                $f, 
                [$(data...)]
            )
        end,
        :whereparams => ()
    )
    return MacroTools.combinedef(def_construct)
end
"""
Replace statements of the form `x ~ Dist(...)` with `x = rand(rng, Dist(...))`
"""
function _replace_sampling(expr)
    expr = MacroTools.postwalk(expr) do sub_expr
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            return :($var = rand(rng, $dist))
        else
            return sub_expr
        end
    end
    return expr
end
"""
Collect all variables that are being sampled. That is, all those variables that
are on the LHS of a `~` statement
"""
function _collect_vars(body)
    vars = []
    MacroTools.postwalk(body) do sub_expr
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            push!(vars, var)
        end
        return sub_expr
    end
    return vars
end
"""
replace all variabls in `vars` with `new_state[var]`
"""
function _replace_variables(expr, vars)
    expr = MacroTools.postwalk(expr) do sub_expr
        if sub_expr in vars
            return :(new_state[$(Meta.quot(sub_expr))])
        else
            return sub_expr
        end
    end
    return expr
end

end

using StableRNGs
using Test

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
