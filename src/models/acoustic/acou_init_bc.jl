
init_bdc!(model::AcousticWaveSimulation, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::AcousticWaveSimulation, srcs::Sources) = nothing

function init_bdc!(
    ::CPMLBoundaryCondition,
    model::AcousticWaveSimulation,
    srcs::Sources
)
    N = length(model.cpmlcoeffs)
    for n in 1:N
        compute_CPML_coefficientsAxis!(
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
        lastcoe = model.cpmlcoeffs[N]

        lastcoe.a[1:length(lastcoe.a)÷2] .= 0.0
        lastcoe.a_h[1:length(lastcoe.a_h)÷2] .= 0.0
        lastcoe.b[1:length(lastcoe.b)÷2] .= 1.0
        lastcoe.b_h[1:length(lastcoe.b_h)÷2] .= 1.0
    end
end
