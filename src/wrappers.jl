
#######################################################

@doc raw"""
    swforward!(
        params::InputParameters{N},
        matprop::MaterialProperties{N}
        shots::Vector{<:Shot} ;  
        parall::Symbol= :threads,
        snapevery::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    )::Union{Vector{AbstractArray}, Nothing} where {N}

Compute forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend if set to `:GPU`
    - `Base.Threads` CPU threads if set to `:threads`
    - otherwise the serial version if set to `:serial`
- `snapevery::Union{Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays.  
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function swforward!(
    params::InputParameters{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    parall::Symbol=:threads,
    snapevery::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing
)::Union{Vector{AbstractArray}, Nothing} where {N}
    # Build wavesim
    wavesim = build_wavesim(params; parall=parall, snapevery=snapevery, infoevery=infoevery, gradient=false)
    # Solve simulation
    return run_swforward!(wavesim, matprop, shots)
end

swforward!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}, shots::Vector{<:Shot}; kwargs...) where {N} =
    run_swforward!(wavesim, matprop, shots; kwargs...)

#######################################################

@doc raw"""
    swmisfit!(
        params::InputParameters{N},
        matprop::MaterialProperties{N},
        shots::Vector{<:Shot} ;  
        parall::Symbol= :threads,
    )::Real where {N}

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.

# Keyword arguments
`parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend if set to `:GPU`
    - `Base.Threads` CPU threads if set to `:threads`
    - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref).
"""
function swmisfit!(
    params::InputParameters{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};  #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol=:threads
)::Real where {N}
    # Build wavesim
    wavesim = build_wavesim(params; parall=parall, gradient=false)
    # Compute misfit
    return run_swmisfit!(wavesim, matprop, shots)
end

swmisfit!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}, shots::Vector{<:Shot}; kwargs...) where {N} =
    run_swmisfit!(wavesim, matprop, shots; kwargs...)

#######################################################

@doc raw"""
    swgradient!(
        params::InputParameters{N},
        matprop::MaterialProperties{N},
        shots::Vector{<:Shot} ;
        parall::Symbol = :threads,
        check_freq::Union{Int, Nothing} = nothing,
        infoevery::Union{Int, Nothing} = nothing
    ):Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}

Compute gradients w.r.t. model parameters using the given input parameters `params` and material parameters `matprop` on multiple shots.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend if set to `:GPU`
    - `Base.Threads` CPU threads if set to `:threads`
    - otherwise the serial version if set to `:serial`
- `check_freq::Union{Int, Nothing} = nothing`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `compute_misfit::Bool = false`: if true, also computes and return misfit value.
"""
function swgradient!(
    params::InputParameters{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot}; #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol=:threads,
    check_freq::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing,
    compute_misfit::Bool=false
)::Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}
    # Build wavesim
    wavesim = build_wavesim(params; parall=parall, infoevery=infoevery, gradient=true, check_freq=check_freq)
    # Solve simulation
    return run_swgradient!(wavesim, matprop, shots; compute_misfit=compute_misfit)
end

@doc raw"""
    swgradient!(wavesim::WaveSimul{N},
                matprop::MaterialProperties{N},
                shots::Vector{<:Shot})

    Compute gradients w.r.t. model parameters using the *previously* built WaveSimul. This avoids re-initializing and re-allocating several arrays in case of multiple gradient calculations.
"""
swgradient!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}, shots::Vector{<:Shot}; kwargs...) where {N} =
    run_swgradient!(wavesim, matprop, shots; kwargs...)

#######################################################

@doc raw"""
    build_wavesim(params::InputParameters{N}, kwargs...)::WaveSimul{N} where {N}

Builds a wave similation based on the input paramters `params` and keyword arguments `kwargs`.

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend if set to `:GPU`
    - `Base.Threads` CPU threads if set to `:threads`
    - otherwise the serial version if set to `:serial`
- `gradient::Bool = false`: whether the wave simulation is used for gradients computations.
- `check_freq::Union{<:Integer, Nothing} = nothing`: if `gradient = true` and if specified, enables checkpointing and specifies the checkpointing frequency.
- `snapevery::Union{<:Integer, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays (only for forward simulations).
- `infoevery::Union{<:Integer, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
build_wavesim(params::InputParameters; kwargs...) = build_wavesim(params, params.boundcond; kwargs...)

build_wavesim(
    params::InputParametersAcoustic{N},
    cpmlparams::CPMLBoundaryConditionParameters;
    kwargs...
) where {N} = AcousticCDCPMLWaveSimul{N}(
    params.gridsize,
    params.gridspacing,
    params.ntimesteps,
    params.dt,
    cpmlparams.halo,
    cpmlparams.rcoef;
    freetop=cpmlparams.freeboundtop,
    kwargs...
)

#######################################################

select_backend(wavesim_type::Type{<:WaveSimul}, parall::Symbol) =
    select_backend(BoundaryConditionTrait(wavesim_type), GridTrait(wavesim_type), wavesim_type, Val{parall})

function select_backend(
    ::BoundaryConditionTrait,
    ::GridTrait,
    wavesim_type::Type{<:WaveSimul},
    ::Type{Val{parall}}
) where {parall}
    parasym = [:serial, :threads, :GPU]
    error(
        "No backend found for model of type $(wavesim_type) and `parall` $(parall). Argument `parall` must be one of the following symbols: $parasym."
    )
end

# Backend selections for AcousticCDCPMLWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:serial}}) =
    Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:serial}}) =
    Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:serial}}) =
    Acoustic3D_CD_CPML_Serial

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:threads}}) =
    Acoustic1D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:threads}}) =
    Acoustic2D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:threads}}) =
    Acoustic3D_CD_CPML_Threads

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:GPU}}) =
    Acoustic1D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:GPU}}) =
    Acoustic2D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:GPU}}) =
    Acoustic3D_CD_CPML_GPU
