
init_bdc!(model::ElasticWaveSimulation, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::ElasticWaveSimulation, srcs::Sources) = nothing

@views function init_bdc!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoWaveSimulation,
    srcs::Sources
)
    N = length(model.cpmlcoeffs)
    for n in 1:N
        #@show n,typeof(model.cpmlcoeffs[n])
        compute_CPML_coefficientsAxis!(
            model.cpmlcoeffs[n],
            get_maximum_func(model)(sqrt.((model.matprop.λ + 2.0 * model.matprop.μ) ./ model.matprop.ρ)),
            model.dt,
            model.cpmlparams.halo,
            model.cpmlparams.rcoef,
            model.grid.spacing[n] * model.cpmlparams.halo,  ##  <<-- Check: dx*(n-1)?
            srcs.domfreq
        )
    end

    if model.cpmlparams.freeboundtop && N >= 1
        lastcoe = model.cpmlcoeffs[N]

        lastcoe.a[1:length(lastcoe.a)÷2] .= 0.0
        lastcoe.a_h[1:length(lastcoe.a_h)÷2] .= 0.0
        lastcoe.b[1:length(lastcoe.b)÷2] .= 1.0
        lastcoe.b_h[1:length(lastcoe.b_h)÷2] .= 1.0
        # model.cpmlcoeffs[N].a_l .= 0.0
        # model.cpmlcoeffs[N].a_hl .= 0.0
        # model.cpmlcoeffs[N].b_l .= 1.0
        # model.cpmlcoeffs[N].b_hl .= 1.0
    end
end
