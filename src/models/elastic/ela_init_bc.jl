
init_bdc!(model::ElasticWaveSimulation, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::ElasticWaveSimulation, srcs::Sources) = nothing

function init_bdc!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoWaveSimulation{T, N},
    srcs::Sources
    ) where {T, N}
    
    if model.cpmlparams.vel_max==nothing
        vel_max = get_maximum_func(model)(sqrt.((model.matprop.λ .+ 2 .* model.matprop.μ) ./ model.matprop.ρ))
    else
        vel_max = model.cpmlparams.vel_max
    end

    for n in 1:N
        #@show n,typeof(model.cpmlcoeffs[n])
        compute_CPML_coefficientsAxis!(
            model.cpmlcoeffs[n],
            vel_max,
            model.dt,
            model.cpmlparams.halo,
            model.cpmlparams.rcoef,
            model.grid.spacing[n] * model.cpmlparams.halo,  ##  <<-- Check: dx*(n-1)?
            srcs.domfreq
        )
    end

    if model.cpmlparams.freeboundtop && N >= 1
        lastcoe = model.cpmlcoeffs[N]

        lastcoe.a[1:length(lastcoe.a)÷2] .= zero(T)
        lastcoe.a_h[1:length(lastcoe.a_h)÷2] .= zero(T)
        lastcoe.b[1:length(lastcoe.b)÷2] .= one(T)
        lastcoe.b_h[1:length(lastcoe.b_h)÷2] .= one(T)
    end
end
