### FORWARDS ###

#######################################################

@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using the given input parameters `params` and material properties `matprop` on multiple shots.
Receivers traces are stored in the receivers for each shot.

Return a vector of `Dict` containing for each shot the snapshots of the fields computed in the simulation for each timestep.

# Positional arguments
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation type.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation type.
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `parall::Symbol = :threads`: controls which backend is used for computation:
    - the `CUDA.jl` GPU backend performing automatic domain decomposition if set to `:CUDA`
    - the `AMDGPU.jl` GPU backend performing automatic domain decomposition if set to `:AMDGPU`
    - the `Metal.jl` GPU backend performing automatic domain decomposition if set to `:Metal`
    - `Base.Threads` CPU threads performing automatic domain decomposition if set to `:threads`
    - `Base.Threads` CPU threads sending a group of sources to each thread if set to `:threadpersrc`
    - otherwise the serial version if set to `:serial`
- `snapevery::Union{Int, Nothing} = nothing`: if specified, saves itermediate snapshots at the specified frequency (one every `snapevery` time step iteration) and return them as a vector of arrays.  
- `infoevery::Union{Int, Nothing} = nothing`: if specified, logs info about the current state of simulation every `infoevery` time steps.
- `logger::Union{Nothing, AbstractLogger} = nothing`: if specified, uses the given `logger` object to print logs, otherwise it uses the logger returned from `current_logger()`.

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
"""
function swforward!(
    params::InputParameters{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    runparams::RunParameters
    # parall::Symbol=:threads,
    # snapevery::Union{Int, Nothing}=nothing,
    # infoevery::Union{Int, Nothing}=nothing,
    # logger::Union{Nothing, AbstractLogger}=nothing
)::Union{Vector{Dict{Int, Dict{String, <:AbstractField{T}}}}, Nothing} where {T, N}
   
    return with_logger(runparams.logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop;
                                runparams=runparams,
                                gradient=false)
        # Solve simulation
        run_swforward!(wavesim, matprop, shots)
    end
end

@doc """

$(TYPEDSIGNATURES)

Compute forward simulation using a previously constructed `WaveSimulation` object. See also [`build_wavesim`](@ref) on how to build the `WaveSimulation`.
Receivers traces are stored in the receivers for each shot.

Return a vector of `Dict` containing for each shot the snapshots of the fields computed in the simulation for each timestep.

# Positional arguments
- `wavesim`: wave simulation object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation type.
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `logger::Union{Nothing, AbstractLogger} = nothing`: if specified, uses the given `logger` object to print logs, otherwise it uses the logger returned from `current_logger()`.

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
"""
function swforward!(
    wavesim::Union{WaveSimulation{T, N}, Vector{<:WaveSimulation{T, N}}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    #logger::Union{Nothing, AbstractLogger}=nothing,
    #kwargs...
)::Union{Vector{Dict{Int, Dict{String, <:AbstractField{T}}}}, Nothing} where {T, N}
    # if logger === nothing
    #     logger = current_logger()
    # end
    return with_logger(wavesim.runparams.logger) do
        run_swforward!(wavesim, matprop, shots)   #; kwargs...)
    end
end

#######################################################

## single WaveSimulation object
function run_swforward!(
    wavesim::WaveSimulation{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
)::Union{Vector{Dict{Int, Dict{String, <:AbstractField{T}}}}, Nothing} where {T, N}

    @info ">=====  Forward simulation  ======<"

    # Check wavesim consistency
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(wavesim, matprop, shots)

    # Set wavesim material properties
    # Remark: matprop in wavesim are mutated
    @debug "Setting wavesim material properties"
    set_wavesim_matprop!(wavesim, matprop)
    # Now onwards matprop from "outside" should not be used anymore!!!

    # Snapshots setup
    takesnapshots = snapenabled(wavesim)
    if takesnapshots
        snapshots_per_shot = []
    end

    # Shots loop
    for (s, singleshot) in enumerate(shots)
        @info "-- Shot #$s --"
        # Initialize shot
        @debug "Initializing shot"
        init_shot!(wavesim, singleshot)
        # Compute forward solver
        @debug "Forward solver"
        swforward_1shot!(wavesim, singleshot)
        # Save shot's snapshots
        if takesnapshots
            @info "Saving snapshots"
            push!(snapshots_per_shot, deepcopy(wavesim.snapshotter.snapshots))
        end
    end

    if takesnapshots
        return snapshots_per_shot
    end
    return nothing
end

## :threadpersrc, multiple WaveSimulation objects
function run_swforward!(
    wavesim::Vector{<:WaveSimulation{T, N}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
)::Union{Vector{Dict{Int, Dict{String, <:AbstractField{T}}}, Nothing}} where {T, N}
    nwsim = length(wavesim)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert nthr == nwsim

    @info ">=====  Forward simulation  ======<"

    snapshots_per_shot = Dict{Int, Vector}()
    for w in 1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(wavesim[w], matprop, shots)
        # Set wavesim material properties
        # Remark: matprop in wavesim are mutated
        @debug "Setting wavesim material properties"
        set_wavesim_matprop!(wavesim[w], matprop)
        # Now onwards matprop from outside should not be used anymore!!!

        # Snapshots setup
        if snapenabled(wavesim[w])
            snapshots_per_shot[w] = []
        end
    end

    # Shots loop
    nshots = length(shots)
    grpshots = distribsrcs(nshots, nthr) # a vector of UnitRange 
    # loop on the set of WaveSimulation
    Threads.@threads for w in 1:nwsim
        # loop on the subset of shots per each WaveSimulation 
        for s in grpshots[w]
            @info "-- Shot #$s --"
            singleshot = shots[s]
            # Initialize shot
            @debug "Initializing shot"
            init_shot!(wavesim[w], singleshot)
            # Compute forward solver
            @debug "Forward solver"
            swforward_1shot!(wavesim[w], singleshot)
            # Save shot's snapshots
            if snapenabled(wavesim[w])
                @info "Saving snapshots"
                push!(snapshots_per_shot[w], deepcopy(wavesim.snapshotter.snapshots))
            end
        end
    end

    if takesnapshots
        return collect(values(snapshots_per_shot))
    end
    return nothing
end
