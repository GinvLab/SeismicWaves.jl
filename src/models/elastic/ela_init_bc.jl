
init_bdc!(model::ElasticWaveSimul, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::ElasticWaveSimul, srcs::Sources) = nothing

@views function init_bdc!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoWaveSimul,
    srcs::Sources
)
    N = length(model.cpmlcoeffs)
    for n in 1:N
        compute_CPML_coefficients!(
            model.cpmlcoeffs[n],
            get_maximum_func(model)( sqrt.((model.matprop.λ+2.0*model.matprop.μ)./model.matprop.ρ) ),
            model.dt,
            model.halo,
            model.rcoef,
            model.gridspacing[n] * model.halo,  ##  <<-- Check: dx*(n-1)?
            srcs.domfreq
        )
    end

    if model.freetop && N >= 1
        model.cpmlcoeffs[N].a_l .= 0.0
        model.cpmlcoeffs[N].a_hl .= 0.0
        model.cpmlcoeffs[N].b_l .= 1.0
        model.cpmlcoeffs[N].b_hl .= 1.0
    end
end
