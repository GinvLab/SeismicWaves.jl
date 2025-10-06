
init_bdc!(model::AcousticWaveSimulation, srcs::Sources) =
    init_bdc!(BoundaryConditionTrait(model), model, srcs)

init_bdc!(::ReflectiveBoundaryCondition, model::AcousticWaveSimulation, srcs::Sources) = nothing

function init_bdc!(
    ::CPMLBoundaryCondition,
    model::AcousticWaveSimulation{T},
    srcs::Sources
    ) where {T}

      if model.cpmlparams.vel_max == nothing
        vel_max = get_maximum_func(model)(model.matprop.vp)
    else
        vel_max = model.cpml_vel_max
      end
    
    N = length(model.cpmlcoeffs)
    for n in 1:N
        compute_CPML_coefficientsAxis!(
            model.cpmlcoeffs[n],
            vel_max,
            model.dt,
            model.cpmlparams.halo,
            model.cpmlparams.rcoef,
            model.grid.spacing[n] * model.cpmlparams.halo,
            srcs.domfreq
        )
    end

    if model.cpmlparams.freeboundtop && N >= 1
        lastcoe = model.cpmlcoeffs[N]

        lastcoe.a[1:length(lastcoe.a)÷2] .= convert(T, 0.0)
        lastcoe.a_h[1:length(lastcoe.a_h)÷2] .= convert(T, 0.0)
        lastcoe.b[1:length(lastcoe.b)÷2] .= convert(T, 1.0)
        lastcoe.b_h[1:length(lastcoe.b_h)÷2] .= convert(T, 1.0)
    end
end
