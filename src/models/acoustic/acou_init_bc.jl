
init_bdc!(model::AcousticWaveSimulation, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::AcousticWaveSimulation, srcs::Sources) = nothing

@views function init_bdc!(
    ::CPMLBoundaryCondition,
    model::AcousticWaveSimulation,
    srcs::Sources
)
    N = length(model.cpmlcoeffs)
    for n in 1:N
        compute_CPML_coefficients!(
            model.cpmlcoeffs[n],
            get_maximum_func(model)(model.matprop.vp),
            model.dt,
            model.cpmlparams.halo,
            model.cpmlparams.rcoef,
            model.grid.spacing[n] * model.cpmlparams.halo,
            srcs.domfreq
        )
    end

    if model.cpmlparams.freeboundtop && N >= 1
        model.cpmlcoeffs[N].a_l .= 0.0
        model.cpmlcoeffs[N].a_hl .= 0.0
        model.cpmlcoeffs[N].b_l .= 1.0
        model.cpmlcoeffs[N].b_hl .= 1.0
    end
end
