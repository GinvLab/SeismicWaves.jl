### UPDATE MATERIAL PROPERTIES ##

@views function set_wavesim_matprop!(wavesim::WaveSimulation{T, N}, matprop::MaterialProperties{T, N}) where {T, N}
    @debug "Checking new material properties"
    check_matprop(wavesim, matprop)
    @debug "Updating WaveSimulation material properties"
    update_matprop!(wavesim, matprop)
end

### FORWARDS ###

## single WaveSimulation object
@views function run_swforward!(
    model::WaveSimulation{T, N},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
)::Union{Vector{Array{T}}, Nothing} where {T, N}

    # Check wavesim consistency
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(model, matprop, shots)

    # Set wavesim material properties
    # Remark: matprop in wavesim are mutated
    @debug "Setting wavesim material properties"
    set_wavesim_matprop!(model, matprop)
    # Now onwards matprop from "outside" should not be used anymore!!!

    # Snapshots setup
    takesnapshots = snapenabled(model)
    if takesnapshots
        snapshots_per_shot = []
    end

    # Shots loop
    for (s, singleshot) in enumerate(shots)
        @info "Shot #$s"
        # Initialize shot
        @debug "Initializing shot"
        init_shot!(model, singleshot)
        # Compute forward solver
        @info "Computing forward solver"
        swforward_1shot!(model, singleshot)
        # Save shot's snapshots
        if takesnapshots
            @info "Saving snapshots"
            push!(snapshots_per_shot, Array(model.snapshots))
        end
    end

    if takesnapshots
        return snapshots_per_shot
    end
    return nothing
end

## :threadpersrc, multiple WaveSimulation objects
@views function run_swforward!(
    model::Vector{<:WaveSimulation{T, N}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
)::Union{Vector{Array{T}}, Nothing} where {T, N}
    nwsim = length(model)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert nthr == nwsim

    snapshots_per_shot = Dict{Int, Vector}()
    for w in 1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(model[w], matprop, shots)
        # Set wavesim material properties
        # Remark: matprop in wavesim are mutated
        @debug "Setting wavesim material properties"
        set_wavesim_matprop!(model[w], matprop)
        # Now onwards matprop from outside should not be used anymore!!!

        # Snapshots setup
        if snapenabled(model[w])
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
            @info "Shot #$s"
            singleshot = shots[s]
            # Initialize shot
            @debug "Initializing shot"
            init_shot!(model[w], singleshot)
            # Compute forward solver
            @info "Computing forward solver"
            swforward_1shot!(model[w], singleshot)
            # Save shot's snapshots
            if snapenabled(model[w])
                @info "Saving snapshots"
                push!(snapshots_per_shot[w], copy(model[w].snapshots))
            end
        end
    end

    if takesnapshots
        return collect(values(snapshots_per_shot))
    end
    return nothing
end

### MISFITS ###

## single or multiple WaveSimulation objects
@views function run_swmisfit!(
    model::Union{WaveSimulation{T, N}, Vector{<:WaveSimulation{T, N}}},
    matprop::MaterialProperties{T, N},
    shots::Vector{<:Shot{T}};
    misfit::AbstractMisfit=L2Misfit(nothing)
)::T where {T, N}

    # Solve forward model for all shots
    run_swforward!(model, matprop, shots)
    # Compute total misfit for all shots
    @info "Computing misfit"
    totmisfitval = 0
    for singleshot in shots
        @debug "Checking invcov matrix"
        if typeof(model) <: Vector{<:WaveSimulation}
            for i in eachindex(model)
                check_invcov_matrix(model[i], singleshot.recs.invcov)
            end
        else
            check_invcov_matrix(model, singleshot.recs.invcov)
        end
        totmisfitval += misfit(singleshot.recs, matprop)
    end

    return totmisfitval
end

### GRADIENTS ###

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
