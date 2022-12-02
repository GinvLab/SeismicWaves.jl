"""
    @views solve!(
        model::WaveModel,
        shots::Vector{Pair{Sources{<:Real}, Receivers{<:Real}}}
    )

Solve the wave propagation equation specified by `WaveModel` on multiple shots.
"""
@views function solve!(
    model::WaveModel,
    shots::Vector{Pair{Sources{<:Real}, Receivers{<:Real}}}
)
    ## TODO (checks, initializations, ...)

    ## TODO distribute shots
    for (srcs, recs) in shots
        forward!(model, srcs, recs)
    end
    ## TODO gather results

end