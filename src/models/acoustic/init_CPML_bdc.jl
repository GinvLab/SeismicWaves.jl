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

function init_CPML_bdc!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel2D,
    srcs::Sources{<:Real}
)
    compute_CPML_coefficients!(
        model.cpmlcoeffs_x,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dx * model.halo,
        srcs.freqdomain
    )
    compute_CPML_coefficients!(
        model.cpmlcoeffs_y,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dy * model.halo,
        srcs.freqdomain
    )
    if model.freetop
        model.cpmlcoeffs_y.a_l .= 0.0
        model.cpmlcoeffs_y.a_hl .= 0.0
        model.cpmlcoeffs_y.b_K_l .= 1.0
        model.cpmlcoeffs_y.b_K_hl .= 1.0
    end
end

function init_CPML_bdc!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel3D,
    srcs::Sources{<:Real}
)
    compute_CPML_coefficients!(
        model.cpmlcoeffs_x,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dx * model.halo,
        srcs.freqdomain
    )
    compute_CPML_coefficients!(
        model.cpmlcoeffs_y,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dy * model.halo,
        srcs.freqdomain
    )
    if model.freetop
        model.cpmlcoeffs_y.a_l .= 0.0
        model.cpmlcoeffs_y.a_hl .= 0.0
        model.cpmlcoeffs_y.b_K_l .= 1.0
        model.cpmlcoeffs_y.b_K_hl .= 1.0
    end
    compute_CPML_coefficients!(
        model.cpmlcoeffs_z,
        maximum(model.vel),
        model.dt,
        model.halo,
        model.rcoef,
        model.dz * model.halo,
        srcs.freqdomain
    )
end