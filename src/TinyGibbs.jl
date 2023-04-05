module TinyGibbs

export @tiny_gibbs, TinyGibbsModel, TinyGibbsSampler

using AbstractMCMC
using Random
using MacroTools
using Distributions
import StatsBase
using MCMCChains

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

_vec_or_scalar(x::Union{AbstractArray, Real}) = isa(x, AbstractArray) ? vec(x) : x
function _make_chain_array(samples::Vector{<:Dict})
    nsamples = length(samples)
    nvars = sum(length.(values(samples[1])))
    types = eltype.(values(samples[1]))
    # We need to somehow make sure that we always get the values in the correct
    # order. This is not guaranteed by `values`.
    k = collect(keys(samples[1]))
    if !all(types .== types[1])
        throw(ErrorException("Types of all elements in the chain must be the same"))
    end
    chain = Array{types[1]}(undef, nsamples, nvars)
    for s in 1:nsamples
        chain[s, :] .= vcat(_vec_or_scalar.([samples[s][kk] for kk in k])...)
    end
    return chain
end
function _make_array_symbols(key::Symbol, val::AbstractArray)
    attach = [string(ci.I) for ci in CartesianIndices(val)]
    attach = replace.(attach, "(" => "[", ")" => "]")
    attach = vec(attach)
    return Symbol.(string(key) .* attach)
end
function _make_chain_symbols(sample::Dict)
    symbols = Symbol[]
    for (key, val) in sample
        if isa(val, AbstractArray)
            symbols = vcat(symbols, _make_array_symbols(key, val))
        else
            push!(symbols, key)
        end
    end
    return symbols
end
function AbstractMCMC.bundle_samples(
    samples, model::TinyGibbsModel, ::TinyGibbsSampler, ::Any, ::Type{Chains}; kwargs...
)
    # Chain is of dimensions samples×var×chains
    if isa(samples[1], Dict)
        array = _make_chain_array(samples)
        symbols = _make_chain_symbols(samples[1])
        return MCMCChains.Chains(array, symbols)
    else
        throw(ErrorException("Samples are neither a Vector{<:Dict} nor a Vector{Vector{<:Dict}}"))
    end
end

function AbstractMCMC.bundle_samples(
    samples, model::TinyGibbsModel, sampler::TinyGibbsSampler, ::Any, ::Type{Dict}; kwargs...
)
    chain = Dict()
    if isa(samples[1], Dict)
        symbols = keys(samples[1])
        for s in symbols
            type = eltype(samples[1][s])
            nd = ndims(samples[1][s])
            vals = Array{type}(undef, (size(samples[1][s])..., length(samples)))
            for i in eachindex(samples)
                selectdim(vals, nd+1, i) .= samples[i][s]
            end
            chain[s] = vals
        end
    else
        throw(ErrorException("Samples are neither a Vector{<:Dict} nor a Vector{Vector{<:Dict}}"))
    end

    return chain
end
function AbstractMCMC.chainsstack(c::Vector{<:Dict})
    chain = Dict()
    symbols = keys(c[1])
    for s in symbols 
        vals = [c[i][s] for i in eachindex(c)]
        nd = ndims(c[1][s])
        vals = cat(vals...; dims=nd+1)
        chain[s] = vals
    end
    return chain
end

end # module