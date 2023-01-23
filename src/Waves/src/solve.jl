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
    check(model)
    # Precompute constant values
    precompute!(model)

    # Snapshots setup
    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    # Shots loop
    for (srcs, recs) in shots
        # Initialize shot
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        # Compute forward solver
        forward!(model, possrcs, posrecs, srctf, traces, backend)
        # Save traces in reeivers seismograms
        copyto!(recs.seismograms, traces)
        # Save shot's snapshots
        if snapenabled(model)
            push!(snapshots_per_shot, copy(model.snapshots))
        end
    end

    if snapenabled(model)
        return snapshots_per_shot
    end
    return nothing
end
