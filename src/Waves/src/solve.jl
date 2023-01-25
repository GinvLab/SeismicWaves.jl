"""
Solve the wave propagation equation specified by `WaveModel` on multiple shots.

Also returns snapshots for every shot if the model has snapshotting enabled.
"""
@views function solve!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    backend
    )::Union{Vector{Array}, Nothing}
    # Check model
    @info "Checking model"
    check(model)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(model)

    # Snapshots setup
    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        # Compute forward solver
        @info "Computing forward solver"
        forward!(model, possrcs, posrecs, srctf, traces, backend)
        # Save traces in reeivers seismograms
        @info "Saving receivers seismograms"
        copyto!(recs.seismograms, traces)
        # Save shot's snapshots
        if snapenabled(model)
            @info "Saving snapshots"
            push!(snapshots_per_shot, copy(model.snapshots))
        end
    end

    if snapenabled(model)
        return snapshots_per_shot
    end
    return nothing
end

@views function solve_gradient!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    invcov::AbstractMatrix{<:Real},
    backend;
    check_freq::Integer = 100
    )
    # Check model
    @info "Checking model"
    check(model)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(model)

    # Snapshots setup
    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad = gradient!(model, possrcs, posrecs, srctf, traces, recs.observed, invcov, backend; check_freq=check_freq)

        # TODO accumulate total gradient
    end

    # TODO return total gradient
    return nothing
end
