### GRADIENTS ###

@doc """

$(TYPEDSIGNATURES)

Compute gradients w.r.t. model parameters using the given input parameters `params` and material parameters `matprop` on multiple shots.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

# Positional arguments
- `params::InputParameters{T, N}`: input parameters for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., acoustic variable-density).
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `runparams::RunParameters`: a struct containing parameters related to forward calculations. See [`RunParameters`](@ref) for details. In case of a forward simulation, `gradparams` is set to `nothing`.
- `gradparams::Union{GradParameters,Nothing}`: a struct containing parameters related to gradient calculations. See [`GradParameters`](@ref) for details.

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
See also [`swforward!`](@ref) and [`swmisfit!`](@ref) and [`Shot`](@ref).
"""
function swgradient!(
    params::InputParameters{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}},
    misfit::Vector{<:AbstractMisfit};
    runparams::RunParameters=RunParameters(),
    gradparams::GradParameters=GradParameters()
)::Union{Dict{String, Array{T, N}},
    Tuple{Dict{String, Array{T, N}}, T}} where {T, N}

    return with_logger(runparams.logger) do
        # Build wavesim
        wavesim = build_wavesim(params, matprop;  runparams=runparams,
                                gradient=true, gradparams=gradparams)
        # Solve simulation
        run_swgradient!(wavesim, matprop, shots, misfit) 
    end
end

@doc """

$(TYPEDSIGNATURES)

Compute gradients w.r.t. model parameters using the *previously* built `WaveSimulation`. This avoids re-initializing and re-allocating several arrays in case of multiple gradient calculations.
See also [`build_wavesim`](@ref) on how to build the `WaveSimulation`.

The `check_freq` parameter controls the checkpoiting frequency for adjoint computation.
If `nothing`, no checkpointing is performed.
If greater than 2, a checkpoint is saved every `check_freq` time step.
The optimal tradeoff value is `check_freq = sqrt(nt)` where `nt` is the number of time steps of the forward simulation.
Bigger values speed up computation at the cost of using more memory.

# Positional arguments
- `wavesim::Union{WaveSimulation{T,N},Vector{<:WaveSimulation{T,N}}}`: input `WaveSimulation` object containing all required information to run the simulation.
- `matprop::MaterialProperties{T, N}`: material properties for the simulation, where T represents the data type and N represents the number of dimensions. They vary depending on the simulation kind (e.g., Vp only is required for an acoustic constant-density simulation).
- `shots::Vector{<:Shot{T}}`: a vector whose elements are `Shot` structures. Each shot contains information about both source(s) and receiver(s).

# Keyword arguments
- `runparams::RunParameters`: a struct containing parameters related to forward calculations. See [`RunParameters`](@ref) for details. In case of a forward simulation, `gradparams` is set to `nothing`.
- `gradparams::Union{GradParameters,Nothing}`: a struct containing parameters related to gradient calculations. See [`GradParameters`](@ref) for details.

See also [`InputParameters`](@ref), [`MaterialProperties`](@ref) and [`Shot`](@ref).
See also [`swforward!`](@ref) and [`swmisfit!`](@ref) and [`Shot`](@ref).
"""
function swgradient!(wavesim::Union{WaveSimulation{T, N},Vector{<:WaveSimulation{T, N}}},
                     matprop::MaterialProperties{T, N},
                     shots::Vector{<:Shot{T}},
                     misfit::Vector{<:AbstractMisfit};
                     )::Union{Dict{String, Array{T, N}}, Tuple{Dict{String, Array{T, N}}, T}} where {T, N}
    
    return with_logger(wavesim.runparams.logger) do
        run_swgradient!(wavesim, matprop, shots)
    end
end

#######################################################

## single WaveSimulation object
function run_swgradient!(
    wavesim::WaveSimulation{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}},
    misfit::Vector{<:AbstractMisfit};
)::Union{Dict{String, Array{T, N}},
    Tuple{Dict{String, Array{T, N}}, T}} where {T, N}

    @info ">=====  Gradient computation  ======<"

    # Check wavesim consistency
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(wavesim, matprop, shots)

    # Set wavesim material properties
    @debug "Setting wavesim material properties"
    set_wavesim_matprop!(wavesim, matprop)

    # Initialize total gradient and total misfit
    totgrad = init_gradient(wavesim)
    totmisfitval = 0
    # Shots loop
    for s in length(shots)
        singleshot = shots[s]
        singlemisfit = misfit[s]
        @info "Shot #$s"
        # Initialize shot
        @debug "Initializing shot"
        init_shot!(wavesim, singleshot)
        @debug "Checking invcov matrix"
        check_invcov_matrix(wavesim, singlemisfit.invcov)
        # Compute forward solver
        @debug "Computing gradient solver"
        curgrad = swgradient_1shot!(wavesim, singleshot, singlemisfit)
                                    
        # Accumulate gradient
        accumulate_gradient!(totgrad, curgrad, wavesim)
        # Compute misfit if needed
        if wavesim.gradparams.compute_misfit
            @info "Computing misfit"
            totmisfitval += calcmisfit(singlemisfit, singleshot.recs)
        end
    end

    return wavesim.gradparams.compute_misfit ? (totgrad, totmisfitval) : totgrad
end

## :threadpersrc, multiple WaveSimulation objects
function run_swgradient!(
    wavesim::Vector{<:WaveSimulation{T, N}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}},
    misfit::Vector{<:AbstractMisfit};
)::Union{Dict{String, Array{T, N}},Tuple{Dict{String, Array{T, N}}, T}} where {T, N}
    
    nwsim = length(wavesim)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert Threads.nthreads() == nwsim

    @info ">=====  Gradient computation  ======<"

    for w in 1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(wavesim[w], matprop, shots)

        # Set wavesim material properties
        @debug "Setting wavesim material properties"
        set_wavesim_matprop!(wavesim[w], matprop)
    end

    # Initialize total gradient and total misfit
    nshots = length(shots)
    allgrad = Vector{Dict{String, Array{T, N}}}(undef, nshots)

    if wavesim.gradparams.compute_misfit
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
            singlemisfit = misfit[s]
            @info "Shot #$s"
            # Initialize shot
            @debug "Initializing shot"
            init_shot!(wavesim[w], singleshot)
            @debug "Checking invcov matrix"
            check_invcov_matrix(wavesim[w], singleshot.recs.invcov)
            # Compute forward solver
            @debug "Gradient solver"
            curgrad = swgradient_1shot!( wavesim[w], singleshot, singlemisfit)

            # Save gradient
            allgrad[s] = curgrad
            # Compute misfit if needed
            if wavesim.gradparams.compute_misfit
                @info "Computing misfit"
                allmisfitval[s] = calcmisfit(singlemisfit, singleshot.recs)
            end
        end
    end

    # Accumulate gradient and misfit
    totgrad = init_gradient(wavesim)
    for curgrad in allgrad
        accumulate_gradient!(wavesim, totgrad, curgrad)
    end
    if gradparams.compute_misfit
        totmisfitval = sum(allmisfitval)
    end

    return wavesim.gradparams.compute_misfit ? (totgrad, totmisfitval) : totgrad
end
