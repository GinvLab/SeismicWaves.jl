"""
    @views solve!(
        model::WaveModel,
        shots::Vector{Pair{Sources{<:Real}, Receivers{<:Real}}}
    )::Union{Vector{Array}, Nothing}

Solve the wave propagation equation specified by `WaveModel` on multiple shots.

Also returns snapshots for every shot if the model has snapshotting enabled.
"""
@views function solve!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}}
)::Union{Vector{Array}, Nothing}
    check(model)
    precompute!(model)

    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    ## TODO distribute shots
    for (srcs, recs) in shots
        forward!(model, srcs, recs)

        if snapenabled(model)
            push!(snapshots_per_shot, model.shapshots)
        end
    end
    ## TODO gather results

    if snapeneabled(model)
        return snapshots_per_shot
    end
    return nothing
end
