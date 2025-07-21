@doc """

$(TYPEDSIGNATURES)     

Builds a wave similation based on the input paramters `params` and keyword arguments `kwargs`.

# Positional arguments
- `params::InputParameters{T,N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).

# Keyword arguments
- `runparams::RunParameters`: a struct containing parameters related to forward calculations. See [`RunParameters`](@ref) for details. In case of a forward simulation, `gradparams` is set to `nothing`.
- `gradparams::Union{GradParameters,Nothing}`: a struct containing parameters related to gradient calculations. See [`GradParameters`](@ref) for details. 
- `gradient::Bool = false`: whether the wave simulation is used for gradients computations.

"""
function build_wavesim(params::InputParameters{T, N}, matprop::MaterialProperties{T, N};
                       runparams::RunParameters,
                       gradparams::Union{GradParameters,Nothing}=nothing,
                       kwargs...) where {T, N}
    if runparams.parall == :threadpersrc
        nthr = Threads.nthreads()
        wsim = [build_concrete_wavesim(params, matprop, params.boundcond; runparams, gradparams, kwargs...) for _ in 1:nthr]
    else
        wsim = build_concrete_wavesim(params, matprop, params.boundcond; runparams, gradparams, kwargs...)
    end
    return wsim
end

build_concrete_wavesim(
    params::InputParametersAcoustic{T, N},
    matprop::VpAcousticCDMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters{T};
    runparams::RunParameters,
    gradparams::Union{GradParameters,Nothing},
    kwargs...
) where {T, N} = AcousticCDCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    gradparams=gradparams,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersAcoustic{T, N},
    matprop::VpRhoAcousticVDMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters;
    runparams::RunParameters,
    gradparams::Union{GradParameters,Nothing},
    kwargs...
) where {T, N} = AcousticVDStaggeredCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    gradparams=gradparams,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersElastic{T, N},
    matprop::ElasticIsoMaterialProperties{T, N},
    cpmlparams::CPMLBoundaryConditionParameters{T};
    runparams::RunParameters,
    gradparams::Union{GradParameters,Nothing},
    kwargs...
) where {T, N} = ElasticIsoCPMLWaveSimulation(
    params,
    matprop,
    cpmlparams;
    runparams=runparams,
    gradparams=gradparams,
    kwargs...
)

#######################################################
