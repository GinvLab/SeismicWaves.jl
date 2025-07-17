@doc """

$(TYPEDSIGNATURES)     

Builds a wave similation based on the input paramters `params` and keyword arguments `kwargs`.

# Positional arguments
- `params::InputParameters{T,N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - the `Metal.jl` GPU backend performing automatic domain decomposition if set to `:Metal`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `gradient::Bool = false`: whether the wave simulation is used for gradients computations.
- `check_freq::Union{<:Int, Nothing} = nothing`: if `gradient = true` and if specified, enables checkpointing and specifies the checkpointing frequency.
- `snapevery::Union{<:Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays (only for forward simulations).
- `infoevery::Union{<:Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function build_wavesim(params::InputParameters{T, N}, matprop::MaterialProperties{T, N};
                       runparams::RunParameters, kwargs...) where {T, N}
    if runparams.parall == :threadpersrc
        nthr = Threads.nthreads()
        wsim = [build_concrete_wavesim(params, matprop, params.boundcond; runparams, kwargs...) for _ in 1:nthr]
    else
        wsim = build_concrete_wavesim(params, matprop, params.boundcond; runparams, kwargs...)
    end
    return wsim
end

build_concrete_wavesim(
    params::InputParametersAcoustic{T, N},
    matprop::VpAcousticCDMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters{T};
    runparams::RunParameters,
    kwargs...
) where {T, N} = AcousticCDCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersAcoustic{T, N},
    matprop::VpRhoAcousticVDMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters;
    runparams::RunParameters,
    kwargs...
) where {T, N} = AcousticVDStaggeredCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersElastic{T, N},
    matprop::ElasticIsoMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters{T};
    runparams::RunParameters,
    kwargs...
) where {T, N} = ElasticIsoCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    kwargs...
)

#######################################################
