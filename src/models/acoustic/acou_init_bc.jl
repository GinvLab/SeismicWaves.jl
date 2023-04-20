
init_bdc!(model::AcousticWaveSimul, matprop::MaterialProperties, srcs::Sources{<:Real}) =
    init_bdc!(BoundaryConditionTrait(model), model, matprop, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::AcousticWaveSimul, matprop::VpAcousticCDMaterialProperty, srcs::Sources{<:Real}) =
    nothing

function init_bdc!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimul,
    matprop::VpAcousticCDMaterialProperty,
    srcs::Sources{<:Real}
)
    N = length(model.cpmlcoeffs)
    for n in 1:N
        compute_CPML_coefficients!(
            model.cpmlcoeffs[n],
            get_maximum_func(model)(matprop.vp),
            model.dt,
            model.halo,
            model.rcoef,
            model.gridspacing[n] * model.halo,
            srcs.domfreq
        )
    end

    if model.freetop && N >= 1
        model.cpmlcoeffs[N].a_l .= 0.0
        model.cpmlcoeffs[N].a_hl .= 0.0
        model.cpmlcoeffs[N].b_K_l .= 1.0
        model.cpmlcoeffs[N].b_K_hl .= 1.0
    end
end
