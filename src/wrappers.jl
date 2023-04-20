
#######################################################

@doc raw"""
    swforward!(
        params::InputParameters,
        matprop::MaterialProperties
        shots::Vector{<:Shot} ;  
        parall::Symbol= :threads,
        snapevery::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    )::Union{Vector{AbstractArray}, Nothing}

Compute forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Keyword arguments
- `parall::Symbol` = :threads: controls which backend is used for computation:
  - the `CUDA.jl` GPU backend if set to `:GPU`
  - `Base.Threads` CPU threads if set to `:threads`
  - otherwise the serial version if set to `:serial`
- `snapevery::Union{Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays  
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function swforward!(
    params::InputParameters,
    matprop::MaterialProperties,
    shots::Vector{<:Shot};
    parall::Symbol=:threads,
    snapevery::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing
)::Union{Vector{AbstractArray}, Nothing}
    # Build wavesim
    wavesim = build_wavesim(params; snapevery=snapevery, infoevery=infoevery)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Solve simulation
    return run_swforward!(wavesim, matprop, backend, shots)
end

#######################################################

@doc raw"""
    swmisfit!(
        params::InputParameters,
        matprop::MaterialProperties,
        shots::Vector{<:Shot} ;  
        parall::Symbol= :threads,
    )::Real

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.

# Keyword arguments
`parall::Symbol` = :threads: controls which backend is used for computation:
  - the `CUDA.jl` GPU backend if set to `:GPU`
  - `Base.Threads` CPU threads if set to `:threads`
  - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref).
"""
function swmisfit!(
    params::InputParameters,
    matprop::MaterialProperties,
    shots::Vector{<:Shot};  #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol=:threads
)::Real
    # Build wavesim
    wavesim = build_wavesim(params)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Compute misfit
    return run_swmisfit!(wavesim, matprop, backend, shots)
end

#######################################################

@doc raw"""
    swgradient!(
        params::InputParameters,
        matprop::MaterialProperties,
        shots::Vector{<:Shot} ;
        parall::Symbol = :threads,
        check_freq::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    )::AbstractArray

Compute gradients w.r.t. model parameters using the given input parameters `params` and material parameters `matprop` on multiple shots.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Keyword arguments
- `parall::Symbol` = :threads: controls which backend is used for computation:
  - the `CUDA.jl` GPU backend if set to `:GPU`
  - `Base.Threads` CPU threads if set to `:threads`
  - otherwise the serial version if set to `:serial`
- `check_freq::Union{Int, Nothing}`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `compute_misfit::Bool = false`: if true, also computes and return misfit value.
"""
function swgradient!(
    params::InputParameters,
    matprop::MaterialProperties,
    shots::Vector{<:Shot}; #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol=:threads,
    check_freq::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing,
    compute_misfit::Bool=false
)::Union{AbstractArray, Tuple{AbstractArray, Real}}
    # Build wavesim
    wavesim = build_wavesim(params; infoevery=infoevery)
    # Select backend
    backend = select_backend(wavesim, parall)
    # Solve simulation
    return run_swgradient!(wavesim, matprop, backend, shots; check_freq=check_freq, compute_misfit=compute_misfit)
end

#######################################################

build_wavesim(params::InputParametersAcoustic; kwargs...) = build_wavesim(params, params.boundcond; kwargs...)

function build_wavesim(
    params::InputParametersAcoustic,
    cpmlparams::CPMLBoundaryConditionParameters;
    kwargs...
)
    N = length(params.gridsize)

    acoumod = AcousticCDCPMLWaveSimul{N}(
        params.gridsize,
        params.gridspacing,
        params.ntimesteps,
        params.dt,
        cpmlparams.halo,
        cpmlparams.rcoef;
        freetop=cpmlparams.freeboundtop,
        kwargs...
    )
    return acoumod
end

#######################################################

select_backend(model::WaveSimul, parall::Symbol) =
    select_backend(BoundaryConditionTrait(model), GridTrait(model), model, Val{parall})

function select_backend(
    ::BoundaryConditionTrait,
    ::GridTrait,
    model::WaveSimul,
    ::Type{Val{parall}}
) where {parall}
    parasym = [:serial, :threads, :GPU]
    error(
        "No backend found for model of type $(typeof(model)) and `parall` $(parall). Argument `parall` must be one of the following symbols: $parasym."
    )
end

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{1}, ::Type{Val{:serial}}) = Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{2}, ::Type{Val{:serial}}) = Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{3}, ::Type{Val{:serial}}) = Acoustic3D_CD_CPML_Serial

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{1}, ::Type{Val{:threads}}) = Acoustic1D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{2}, ::Type{Val{:threads}}) = Acoustic2D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{3}, ::Type{Val{:threads}}) = Acoustic3D_CD_CPML_Threads

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{1}, ::Type{Val{:GPU}}) = Acoustic1D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{2}, ::Type{Val{:GPU}}) = Acoustic2D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::AcousticCDWaveSimul{3}, ::Type{Val{:GPU}}) = Acoustic3D_CD_CPML_GPU
