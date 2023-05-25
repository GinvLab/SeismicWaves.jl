swforward_1shot!(model::ElasticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoWaveSimul{N},
    possrcs,
    posrecs,
    srctf,
    traces
) where {N}
    # Velocity arrays
    vx = model.vx
    vz = model.vz
    # Stress arrays
    σxx = model.σxx
    σzz = model.σzz
    σxz = model.σxz
    # C-PML memory arrays
    
    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_a = model.backend.Data.Array(possrcs)
    posrecs_a = model.backend.Data.Array(posrecs)
    srctf_a = model.backend.Data.Array(srctf)
    traces_a = model.backend.Data.Array(traces)
    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        model.backend.forward_onestep_CPML!(model.vx, model.vz,
                                            model.σxx, model.σzz, model.σxz,
                                            model.ρ, model.ρ_ihalf_jhalf,
                                            model.μ, model.μ_ihalf, model.μ_jhalf,
                                            model.dt,
                                            model.ψ_∂σxx∂x, model.ψ_∂σxz∂z,
                                            model.ψ_∂σxz∂x, model.ψ_∂σzz∂z,
                                            model.ψ_∂vx∂x, model.ψ_∂vz∂z,
                                            model.ψ_∂vx∂z, model.ψ_∂vz∂x,
                                            model.a_coeffs..., model.b_coeffs...,
                                            possrcs_a, srctf_a, posrecs_a, traces_a, it,
                                            freetop,save_trace)

        # pold, pcur, pnew = model.backend.forward_onestep_CPML!(
        #     pold, pcur, pnew, model.fact,
        #     model.gridspacing..., model.halo,
        #     model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
        #     possrcs_a, srctf_a, posrecs_a, traces_a, it
        # )




        
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                model.dt * (it - 1),
                maximum(abs.(Array(pcur)))
            )
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)] .= Array(pcur)
        end
    end

    # Save traces
    traces .= Array(traces_a)
end
