function init_CPML_bdc!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel,
    srcs::Sources{<:Real}
)
    N = length(model.cpmlcoeffs)
    for n = 1:N
        compute_CPML_coefficients!(
            model.cpmlcoeffs[n],
            get_maximum_func(model)(model.vel),
            model.dt,
            model.halo,
            model.rcoef,
            model.Δs[n] * model.halo,
            srcs.freqdomain
        )
    end

    if model.freetop && N > 1
        model.cpmlcoeffs[N].a_l .= 0.0
        model.cpmlcoeffs[N].a_hl .= 0.0
        model.cpmlcoeffs[N].b_K_l .= 1.0
        model.cpmlcoeffs[N].b_K_hl .= 1.0
    end
end