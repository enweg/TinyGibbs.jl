module TinyGibbs

export @tiny_gibbs, TinyGibbsModel, TinyGibbsSampler

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

end # module