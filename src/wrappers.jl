
#######################################################

@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the `Receivers` object for each shot. See also [`Receivers`](@ref).

Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Positional arguments
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
    params::InputParameters{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot};
    parall::Symbol=:threads,
    snapevery::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing,
    logger::Union{Nothing, AbstractLogger}=nothing
)::Union{Vector{Array{T, N}}, Nothing} where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop; parall=parall, snapevery=snapevery, infoevery=infoevery, gradient=false)
        # Solve simulation
        run_swforward!(wavesim, matprop, shots)
    end
end

@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using a previously constructed `WaveSimulation` object.
Return a vector of snapshots for every shot if snapshotting is enabled.

See also [`Sources`](@ref), [`Receivers`](@ref).

# Positional arguments
- `wavesim::Union{WaveSimulation{T,N},Vector{<:WaveSimulation{T,N}}}`: input `WaveSimulation` object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
function swforward!(wavesim::Union{WaveSimulation{T,N}, Vector{<:WaveSimulation{T,N}}}, matprop::MaterialProperties{T, N}, shots::Vector{<:Shot};
    logger::Union{Nothing, AbstractLogger}=nothing, kwargs...)::Union{Vector{Array{T, N}}, Nothing} where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        run_swforward!(wavesim, matprop, shots; kwargs...)
    end
end

#######################################################

@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.

# Positional arguments
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
    params::InputParameters{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot};
    parall::Symbol=:threads,
    misfit::AbstractMisfit=L2Misfit(nothing),
    logger::Union{Nothing, AbstractLogger}=nothing
)::T where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop; parall=parall, gradient=false)
        # Compute misfit
        run_swmisfit!(wavesim, matprop, shots; misfit=misfit)
    end
end

@doc """

$(TYPEDSIGNATURES)

Return the misfit w.r.t. observed data by running a forward simulation using the given `WaveSimulation` object as an input.

# Positional arguments
- `wavesim::Union{WaveSimulation{T,N},Vector{<:WaveSimulation{T,N}}}`: input `WaveSimulation` object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
function swmisfit!(wavesim::Union{WaveSimulation{T,N}, Vector{<:WaveSimulation{T,N}}}, matprop::MaterialProperties{T, N}, shots::Vector{<:Shot};
    logger::Union{Nothing, AbstractLogger}=nothing, kwargs...)::T where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        run_swmisfit!(wavesim, matprop, shots; kwargs...)
    end
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
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
- `smooth_radius::Int = 5`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from sources positions.
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used. 
"""
function swgradient!(
    params::InputParameters{T,N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot};
    parall::Symbol=:threads,
    check_freq::Union{Int, Nothing}=nothing,
    infoevery::Union{Int, Nothing}=nothing,
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing),
    smooth_radius::Int=5,
    logger::Union{Nothing, AbstractLogger}=nothing
)::Union{Dict{String, Array{T, N}},
         Tuple{Dict{String, Array{T, N}}, T}} where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop; parall=parall, infoevery=infoevery, gradient=true, check_freq=check_freq, smooth_radius=smooth_radius)
        # Solve simulation
        run_swgradient!(wavesim, matprop, shots; compute_misfit=compute_misfit, misfit=misfit)
    end
end

@doc """

$(TYPEDSIGNATURES)

Compute gradients w.r.t. model parameters using the *previously* built `WaveSimulation`. This avoids re-initializing and re-allocating several arrays in case of multiple gradient calculations.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

See also [`Sources`](@ref), [`Receivers`](@ref), [`swforward!`](@ref), [`swmisfit!`](@ref).

# Positional arguments
- `wavesim::Union{WaveSimulation{T,N},Vector{<:WaveSimulation{T,N}}}`: input `WaveSimulation` object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
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
- `smooth_radius::Int = 5`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from sources positions.
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used. 
"""
function swgradient!(wavesim::Union{WaveSimulation{T,N}, Vector{<:WaveSimulation{T,N}}}, matprop::MaterialProperties{T, N}, shots::Vector{<:Shot};
    logger::Union{Nothing, AbstractLogger}=nothing, kwargs...)::Union{Dict{String, Array{T, N}},
                                                                      Tuple{Dict{String, Array{T, N}}, T}} where {T, N}
    if logger === nothing
        logger = current_logger()
    end
    return with_logger(logger) do
        run_swgradient!(wavesim, matprop, shots; kwargs...)
    end
end

#######################################################

@doc """

$(TYPEDSIGNATURES)     

Builds a wave similation based on the input paramters `params` and keyword arguments `kwargs`.

# Positional arguments
- `params::InputParameters{T,N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:GPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `gradient::Bool = false`: whether the wave simulation is used for gradients computations.
- `check_freq::Union{<:Int, Nothing} = nothing`: if `gradient = true` and if specified, enables checkpointing and specifies the checkpointing frequency.
- `snapevery::Union{<:Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays (only for forward simulations).
- `infoevery::Union{<:Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
"""
function build_wavesim(params::InputParameters{T, N}, matprop::MaterialProperties{T, N}; parall::Symbol, kwargs...) where {T, N}
    if parall == :threadpersrc
        nthr = Threads.nthreads()
        wsim = [build_concrete_wavesim(params, matprop, params.boundcond; parall=parall, kwargs...) for _ in 1:nthr]
    else
        wsim = build_concrete_wavesim(params, matprop, params.boundcond; parall=parall, kwargs...)
    end
    return wsim
end

build_concrete_wavesim(
    params::InputParametersAcoustic{T,N},
    matprop::VpAcousticCDMaterialProperties{T,N},
    cpmlparams::CPMLBoundaryConditionParameters{T};
    kwargs...
) where {T,N} = AcousticCDCPMLWaveSimul(
    params,
    matprop,
    cpmlparams;
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersAcoustic{T,N},
    matprop::VpRhoAcousticVDMaterialProperties{T,N},
    cpmlparams::CPMLBoundaryConditionParameters;
    kwargs...
) where {T,N} = AcousticVDStaggeredCPMLWaveSimul(
    params.gridsize,
    params.gridspacing,
    params.ntimesteps,
    params.dt,
    matprop,
    cpmlparams.halo,
    cpmlparams.rcoef;
    freetop=cpmlparams.freeboundtop,
    kwargs...
)

build_concrete_wavesim(
    params::InputParametersElastic{T,N},
    matprop::ElasticIsoMaterialProperties{T,N},
    cpmlparams::CPMLBoundaryConditionParameters;
    kwargs...
) where {T,N} = ElasticIsoCPMLWaveSimul(
    params.gridsize,
    params.gridspacing,
    params.ntimesteps,
    params.dt,
    matprop,
    cpmlparams.halo,
    cpmlparams.rcoef;
    freetop=cpmlparams.freeboundtop,
    kwargs...
)

#######################################################
