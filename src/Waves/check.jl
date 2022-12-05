"""
    check(model::WaveModel)

Check model for possible assertions based on configuration and model traits.
"""
check(model::WaveModel) = check(WaveEquationTrait(model), model)

check(x::IsotropicAcoustic, model::WaveModel) = check_courant_condition(x, model)


"""
    check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real})

Check shot configuration againts model based on model traits.
"""
check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = check_shot(WaveEquationTrait(model), BoundaryConditionTrait(model), model, srcs, recs)

check_shot(x::IsotropicAcoustic, ::Reflective, model::WaveModel, srcs::Sources{<:Real}, ::Receivers{<:Real}) = check_ppw(x, model, srcs)
