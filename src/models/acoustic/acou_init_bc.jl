
function init_bdc!(
    model::Acoustic_CD_CPML_WaveSimul,
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
    return
end


function init_bdc!(
    ::Acoustic_CD_Refl_WaveSimul,
    ::Sources{<:Real}
    )
    return
end
