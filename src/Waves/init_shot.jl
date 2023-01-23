"""
    @views function init_shot!(
        model::WaveModel,
        srcs::Sources{<:Real},
        recs::Receivers{<:Real}
    )

Initialize the model for a new shot.
"""
@views function init_shot!(
    model::WaveModel,
    srcs::Sources{<:Real},
    recs::Receivers{<:Real}
)
    init_bdc!(model, srcs)
    check_shot(model, srcs, recs)
    return allocate_shot(model, srcs, recs)
end

"""
    check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real})

Check shot configuration for a model.
"""
check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = check_shot(WaveEquationTrait(model), BoundaryConditionTrait(model), model, srcs, recs)

function check_shot(x::IsotropicAcousticWaveEquation, ::ReflectiveBoundaryCondition, model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    @debug "Checking points per wavelengh"
    check_ppw(x, model, srcs)
    @debug "Checking sources positions"
    check_positions(model, srcs.positions)
    @debug "Checking receivers positions" 
    check_positions(model, recs.positions)
end

"""
    init_bdc!(model::WaveModel, srcs::Sources{<:Real})

Initialize model boundary conditions for a shot.
"""
init_bdc!(model::WaveModel, srcs::Sources{<:Real}) = init_bdc!(WaveEquationTrait(model), BoundaryConditionTrait(model), model, srcs)
init_bdc!(::AcousticWaveEquation, ::ReflectiveBoundaryCondition, ::WaveModel, ::Sources{<:Real}) = nothing

allocate_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = allocate_shot(WaveEquationTrait(model), model, srcs, recs)
allocate_shot(x::AcousticWaveEquation, model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = extract_shot(x, model, srcs, recs)
