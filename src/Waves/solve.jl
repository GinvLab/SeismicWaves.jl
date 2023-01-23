"""
Solve the wave propagation equation specified by `WaveModel` on multiple shots.

Also returns snapshots for every shot if the model has snapshotting enabled.
"""
@views function solve!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    backend
)::Union{Vector{Array}, Nothing}
    check(model)
    precompute!(model)

    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    for (srcs, recs) in shots
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        forward!(model, possrcs, posrecs, srctf, traces, backend)
        # Save traces in seismograms
        copyto!(recs.seismograms, traces)

        if snapenabled(model)
            push!(snapshots_per_shot, copy(model.snapshots))
        end
    end

    if snapenabled(model)
        return snapshots_per_shot
    end
    return nothing
end
