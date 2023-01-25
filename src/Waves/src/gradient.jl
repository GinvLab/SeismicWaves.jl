gradient!(model::WaveModel, possrcs, posrecs, srctf, traces, observed, invcov, backend; check_freq=check_freq) = gradient!(
    WaveEquationTrait(model),
    BoundaryConditionTrait(model),
    model, possrcs, posrecs, srctf, traces, observed, invcov, backend; check_freq=check_freq
)