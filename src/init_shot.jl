"""
    @views function init_shot!(
        model::WaveSimul,
        srcs::Sources{<:Real},
        recs::Receivers{<:Real}
    )

Initialize the model for a new shot.
"""
@views function init_shot!(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # Check shot configuration
    check_shot(model, srcs, recs)
    # Initialize boundary conditions based on current shot
    init_bdc!(model, srcs)
    # Return allocated shot's arrays
    return setup_shot(model, srcs, recs)
end

"""
    check_shot(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real})

Check shot configuration for a model.
"""
function check_shot(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    @debug "Checking points per wavelengh"
    check_ppw(model, srcs)
    @debug "Checking sources positions"
    check_positions(model, srcs.positions)
    @debug "Checking receivers positions"
    return check_positions(model, recs.positions)
end
