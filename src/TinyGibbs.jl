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
    # println(def[:body])
    def[:body] = MacroTools.postwalk(def[:body]) do sub_expr
        if MacroTools.@capture(sub_expr, var_ ~ dist_)
            return :($var = new_state[$(Meta.quot(var))] = rand(rng, $dist))
        else
            return sub_expr
        end
    end
    pushfirst!(def[:body].args, :(new_state = deepcopy(state)))
    push!(def[:body].args, :(return new_state))
    def[:args] = vcat([:(rng::Random.AbstractRNG), :(state::Dict)], def[:args])
    return MacroTools.combinedef(def)
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


@tiny_gibbs function m()
    a ~ Normal(b, z^2)
    b ~ Normal(a/2, 1)
    z ~ Gamma(a^2, b^2)
end
using StableRNGs
rng = StableRNG(123)
state = Dict(:a => 1.0, :b => 1.0, :z => 1.0)
state = m(rng, state)


# The above should become
function m(rng, state)
    new_state = deepcopy(state)
    new_state[:a] = rand(rng, Normal(new_state[:b], new_state[:z]^2))
    new_state[:b] = rand(rng, Normal(new_state[:a]/2, 1))
    new_state[:z] = rand(rng, Gamma(new_state[:a]^2, new_state[:b]^2))
    return new_state
end


