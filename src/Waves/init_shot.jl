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
    ## reset!(model)
    ## allocate_shot!(model, srcs, recs)
    ## precompute_shot!(model, srcs, recs)
end

"""
    check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real})

Check shot configuration againts model based on model traits.
"""
check_shot(model::WaveModel, srcs::Sources{<:Real}, recs::Receivers{<:Real}) = check_shot(WaveEquationTrait(model), BoundaryConditionTrait(model), model, srcs, recs)

check_shot(x::IsotropicAcoustic, ::Reflective, model::WaveModel, srcs::Sources{<:Real}, ::Receivers{<:Real}) = check_ppw(x, model, srcs)

"""
    init_bdc!(model::WaveModel, srcs::Sources{<:Real})

Initialize model boundary conditions for a shot.
"""
init_bdc!(model::WaveModel, srcs::Sources{<:Real}) = init_bdc!(WaveEquationTrait(model), BoundaryConditionTrait(model), model, srcs)

init_bdc!(::IsotropicAcoustic, ::Reflective, ::WaveModel, ::Sources{<:Real}) = nothing