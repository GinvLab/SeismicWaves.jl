
#######################################################

@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Positional arguments
- `params::InputParameters{N}`: input parameters for the simulation, where N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{N}`: material properties for the simulation,  where N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
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
    infoevery::Union{Int, Nothing}=nothing,
    logger::Union{Nothing,AbstractLogger}=nothing
)::Union{Vector{AbstractArray}, Nothing} where {N}

    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params; parall=parall, snapevery=snapevery, infoevery=infoevery, gradient=false)
        # Solve simulation
        out = run_swforward!(wavesim, matprop, shots)
    end
    return out
end


@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using a previously constructed `WaveSimul` object.
Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Positional arguments
- `wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}`: input `WaveSimul` object containing all required information to run the simulation.

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `snapevery::Union{Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays.  
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
    """
function swforward!(wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}, matprop::MaterialProperties{N}, shots::Vector{<:Shot};
                    logger::Union{Nothing,AbstractLogger}=nothing, kwargs...) where {N}
    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        out = run_swforward!(wavesim, matprop, shots; kwargs...)
    end
    return out
end

#######################################################

@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.

# Positional arguments
- `params::InputParameters{N}`: input parameters for the simulation, where N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{N}`: material properties for the simulation,  where N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref).
"""
function swmisfit!(
    params::InputParameters{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};  #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    parall::Symbol=:threads,
    misfit::AbstractMisfit=L2Misfit(nothing),
    logger::Union{Nothing,AbstractLogger}=nothing
)::Real where {N}

    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params; parall=parall, gradient=false)
        # Compute misfit
        out = run_swmisfit!(wavesim, matprop, shots; misfit=misfit)
    end
    return out
end


@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given `WaveSimul` object as an input.

# Positional arguments
- `wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}`: input `WaveSimul` object containing all required information to run the simulation.

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`

Receivers traces are stored in the `Receivers` object for each shot.
    
See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref).
"""
function swmisfit!(wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}, matprop::MaterialProperties{N}, shots::Vector{<:Shot}; 
                   logger::Union{Nothing,AbstractLogger}=nothing, kwargs...) where {N}
    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        out = run_swmisfit!(wavesim, matprop, shots; kwargs...)
    end
    return out
end

#######################################################

@doc """

$(TYPEDSIGNATURES)

Compute gradients w.r.t. model parameters using the given input parameters `params` and material parameters `matprop` on multiple shots.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Positional arguments
- `params::InputParameters{N}`: input parameters for the simulation, where N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{N}`: material properties for the simulation,  where N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `check_freq::Union{Int, Nothing} = nothing`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `compute_misfit::Bool = false`: if true, also computes and return misfit value.
- `smooth_radius::Integer = 5`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from sources positions.
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used. 
"""
function swgradient!(
    params::InputParameters{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot}; 
    parall::Symbol=:threads,
    check_freq::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing,
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing),
    smooth_radius::Integer=5,
    logger::Union{Nothing,AbstractLogger}=nothing
)::Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}

    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params; parall=parall, infoevery=infoevery, gradient=true, check_freq=check_freq, smooth_radius=smooth_radius)
        # Solve simulation
        out = run_swgradient!(wavesim, matprop, shots; compute_misfit=compute_misfit, misfit=misfit)
    end
    return out
end


@doc """

$(TYPEDSIGNATURES)

Compute gradients w.r.t. model parameters using the *previously* built `WaveSimul`. This avoids re-initializing and re-allocating several arrays in case of multiple gradient calculations.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Positional arguments
- `wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}`: input `WaveSimul` object containing all required information to run the simulation.

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `check_freq::Union{Int, Nothing} = nothing`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `compute_misfit::Bool = false`: if true, also computes and return misfit value.
- `smooth_radius::Integer = 5`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from sources positions.
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used. 
"""
function swgradient!(wavesim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}}, matprop::MaterialProperties{N}, shots::Vector{<:Shot};
                     logger::Union{Nothing,AbstractLogger}=nothing, kwargs...) where {N} 
    if logger==nothing
        logger=current_logger()
    end
    out = nothing
    with_logger(logger) do
        out = run_swgradient!(wavesim, matprop, shots; kwargs...)
    end
    return out
end
        
#######################################################

@doc """

$(TYPEDSIGNATURES)     

Builds a wave similation based on the input paramters `params` and keyword arguments `kwargs`.

# Positional arguments
- `params::InputParameters{N}`: input parameters for the simulation, where N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `gradient::Bool = false`: whether the wave simulation is used for gradients computations.
- `check_freq::Union{<:Integer, Nothing} = nothing`: if `gradient = true` and if specified, enables checkpointing and specifies the checkpointing frequency.
- `snapevery::Union{<:Integer, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays (only for forward simulations).
- `infoevery::Union{<:Integer, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function build_wavesim(params::InputParameters; parall, kwargs...)
    if parall==:threadpersrc
        nthr = Threads.nthreads()
        #println("  build_wavesim  :threadpersrc")
        wsim = [build_concrete_wavesim(params, params.boundcond; parall, kwargs...) for s=1:nthr]
    else
        wsim = build_concrete_wavesim(params, params.boundcond; parall, kwargs...)
    end
    return wsim
end



build_concrete_wavesim(
    params::InputParametersAcoustic{N},
    cpmlparams::CPMLBoundaryConditionParameters;
    parall,
    kwargs...
) where {N} = AcousticCDCPMLWaveSimul{N}(
    params.gridsize,
    params.gridspacing,
    params.ntimesteps,
    params.dt,
    cpmlparams.halo,
    cpmlparams.rcoef;
    freetop=cpmlparams.freeboundtop,
    parall,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersAcousticVariableDensity{N},
    cpmlparams::CPMLBoundaryConditionParameters;
    parall,
    kwargs...
) where {N} = AcousticVDStaggeredCPMLWaveSimul{N}(
    params.gridsize,
    params.gridspacing,
    params.ntimesteps,
    params.dt,
    cpmlparams.halo,
    cpmlparams.rcoef;
    freetop=cpmlparams.freeboundtop,
    parall,
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
    parasym = [:serial, :threads, :GPU, :threadpersrc]
    error(
        "No backend found for model of type $(wavesim_type) and `parall` $(parall). Argument `parall` must be one of the following symbols: $parasym."
    )
end

## Remark:
## CD = constant density
## VD = variable density

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

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:threadpersrc}}) =
    Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:threadpersrc}}) =
    Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:threadpersrc}}) =
    Acoustic3D_CD_CPML_Serial

# Backend selections for AcousticVDStaggeredCPMLWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{1}}, ::Type{Val{:threads}}) =
    Acoustic1D_VD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{1}}, ::Type{Val{:GPU}}) =
    Acoustic1D_VD_CPML_GPU

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{2}}, ::Type{Val{:threads}}) =
    Acoustic2D_VD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{2}}, ::Type{Val{:GPU}}) =
    Acoustic2D_VD_CPML_GPU
