"""
Compute the forward wave propagation on the specified `WaveModel` and sources/receivers configurations.
The result of the computation is saved in the `WaveModel`.
"""
forward!(model::WaveModel, possrcs, posrecs, srctf, traces, backend) = forward!(
    WaveEquationTrait(model),
    BoundaryConditionTrait(model),
    model, possrcs, posrecs, srctf, traces, backend
)