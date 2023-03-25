"""
    @views function init_shot!(
        model::WaveSimul,
        srcs::Sources{<:Real},
        recs::Receivers{<:Real}
    )

Initialize the model for a new shot.
"""
@views function init_shot!(
    model::WaveSimul,
    srcs::Sources{<:Real},
    recs::Receivers{<:Real}
)
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
    check_positions(model, recs.positions)
end
# check_shot(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = check_shot(model, BoundaryConditionTrait(model), srcs, recs)



# """
#     init_bdc!(model::WaveSimul, srcs::Sources{<:Real})

# Initialize model boundary conditions for a shot.
# """
# init_bdc!(model::WaveSimul, srcs::Sources{<:Real}) = init_bdc!(model, BoundaryConditionTrait(model), model, srcs)
# init_bdc!(::WaveSimul, ::Refl_BC, ::Sources{<:Real}) = nothing
# init_bdc!(model::WaveSimul, ::CPML_BC, srcs::Sources{<:Real}) = init_CPML_bdc!(model, srcs)

# # allocate_shot(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = allocate_shot(WaveEquationTrait(model), model, srcs, recs)


