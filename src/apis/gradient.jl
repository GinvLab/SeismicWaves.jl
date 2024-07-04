### GRADIENTS ###

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
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
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
    shots::Vector{<:Shot{T}};
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
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `check_freq::Union{Int, Nothing} = nothing`: if specified, enables checkpointing and specifies the checkpointing frequency.
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `compute_misfit::Bool = false`: if true, also computes and return misfit value.
- `smooth_radius::Int = 5`: grid points inside a ball with radius specified by the parameter (in grid points) will have their gradient smoothed by a factor inversely proportional to their distance from sources positions.
- `logger::Union{Nothing,AbstractLogger}`: specifies the logger to be used. 
"""
function swgradient!(wavesim::Union{WaveSimulation{T,N}, Vector{<:WaveSimulation{T,N}}}, matprop::MaterialProperties{T, N}, shots::Vector{<:Shot{T}};
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

## single WaveSimulation object
@views function run_swgradient!(
    model::WaveSimulation{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing)
)::Union{Dict{String, Array{T, N}},
    Tuple{Dict{String, Array{T, N}}, T}} where {T, N}

    # Check wavesim consistency
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(model, matprop, shots)

    # Set wavesim material properties
    @debug "Setting wavesim material properties"
    set_wavesim_matprop!(model, matprop)

    # Initialize total gradient and total misfit
    totgrad = init_gradient(model)
    totmisfitval = 0
    # Shots loop
    for (s, singleshot) in enumerate(shots)
        @info "Shot #$s"
        # Initialize shot
        @debug "Initializing shot"
        init_shot!(model, singleshot)
        @debug "Checking invcov matrix"
        check_invcov_matrix(model, singleshot.recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad = swgradient_1shot!(model, singleshot, misfit)
        # Accumulate gradient
        accumulate_gradient!(totgrad, curgrad, model)
        # Compute misfit if needed
        if compute_misfit
            @info "Computing misfit"
            totmisfitval += misfit(singleshot.recs, matprop)
        end
    end

    return compute_misfit ? (totgrad, totmisfitval) : totgrad
end

## :threadpersrc, multiple WaveSimulation objects
@views function run_swgradient!(
    model::Vector{<:WaveSimulation{T, N}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing)
)::Union{Dict{String, Array{T, N}},
    Tuple{Dict{String, Array{T, N}}, T}} where {T, N}
    nwsim = length(model)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert Threads.nthreads() == nwsim

    for w in 1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(model[w], matprop, shots)

        # Set wavesim material properties
        @debug "Setting wavesim material properties"
        set_wavesim_matprop!(model[w], matprop)
    end

    # Initialize total gradient and total misfit
    nshots = length(shots)
    allgrad = Vector{Dict{String, Array{T, N}}}(undef, nshots)

    if compute_misfit
        allmisfitval = zeros(T, nshots)
        totmisfitval = 0
    end

    # Shots loop
    grpshots = distribsrcs(nshots, nthr) # a vector of UnitRange 
    # loop on the set of WaveSimulation
    Threads.@threads for w in 1:nwsim
        # loop on the subset of shots per each WaveSimulation 
        for s in grpshots[w]
            singleshot = shots[s]
            @info "Shot #$s"
            # Initialize shot
            @debug "Initializing shot"
            init_shot!(model[w], singleshot)
            @debug "Checking invcov matrix"
            check_invcov_matrix(model[w], singleshot.recs.invcov)
            # Compute forward solver
            @info "Computing gradient solver"
            curgrad = swgradient_1shot!(
                model[w], singleshot, misfit
            )
            # Save gradient
            allgrad[s] = curgrad
            # Compute misfit if needed
            if compute_misfit
                @info "Computing misfit"
                allmisfitval[s] = misfit(singleshot.recs, matprop)
            end
        end
    end

    # Accumulate gradient and misfit
    totgrad = init_gradient(model)
    for curgrad in allgrad
        accumulate_gradient!(model, totgrad, curgrad)
    end
    if compute_misfit
        totmisfitval = sum(allmisfitval)
    end

    return compute_misfit ? (totgrad, totmisfitval) : totgrad
end
