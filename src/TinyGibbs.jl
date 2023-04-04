module TinyGibbs
using AbstractMCMC
using Random
using MacroTools
using Distributions

# The types here should be Val{GibbsModelName}
struct TinyGibbsSampler <: AbstractMCMC.AbstractSampler 
    draw::Function
end 
struct TinyGibbsModel <: AbstractMCMC.AbstractModel 
    initial_values::NamedTuple
end

# Need a macro that takes in a description of the sampling steps
# and creates a sample method that extends the sample method from 
# AbstractMCMC

function AbstractMCMC.step(
    rng::Random.AbstractRNG, 
    model::TinyGibbsModel, 
    sampler::TinyGibbsSampler; 
    kwargs...
)
    v = sampler.draw(rng, model.initial_values)
    return v, v
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, 
    model::TinyGibbsModel, 
    sampler::TinyGibbsSampler,
    state; 
    kwargs...
)
    v = sampler.draw(rng, state)
    return v, v
end


macro tiny_gibbs(expr)
    return esc(tiny_gibbs(expr))
end
function tiny_gibbs(expr)
    def = MacroTools.splitdef(expr)
    vars = _collect_vars(def[:body])
    def[:body] = MacroTools.postwalk(def[:body]) do sub_expr
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            return :($var = rand(rng, $dist))
        else
            return sub_expr
        end
    end
    def[:body] = _replace_variables(def[:body], vars)
    pushfirst!(def[:body].args, :(new_state = deepcopy(state)))
    push!(def[:body].args, :(return new_state))
    def[:args] = vcat([:(rng::Random.AbstractRNG), :(state::Dict)], def[:args])
    return MacroTools.combinedef(def)
end
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
macro_state = m(StableRNG(123), state)
test_state = m_test(StableRNG(123), state)
@test test_state == macro_state


@tiny_gibbs function m2()
    a ~ Normal(b, z^2)
    b ~ Normal(a/2, 1)
    z ~ Gamma(a^2, b^2)
end
function m2_test(rng, state)
    new_state = deepcopy(state)
    new_state[:a] = rand(rng, Normal(new_state[:b], new_state[:z]^2))
    new_state[:b] = rand(rng, Normal(new_state[:a]/2, 1))
    new_state[:z] = rand(rng, Gamma(new_state[:a]^2, new_state[:b]^2))
    return new_state
end
state = Dict(:a => 1.0, :b => 0.5, :z => 10.0)
macro_state = m2(StableRNG(123), state)
test_state = m2_test(StableRNG(123), state)
@test test_state == macro_state

