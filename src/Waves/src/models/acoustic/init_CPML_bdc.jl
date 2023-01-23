function init_CPML_bdc!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel1D,
    srcs::Sources{<:Real}
)
    compute_CPML_coefficients!(
        model.cpmlcoeffs,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dx * model.halo,
        srcs.freqdomain
    )
end
