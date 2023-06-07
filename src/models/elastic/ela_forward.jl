swforward_1shot!(model::ElasticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoWaveSimul{N},
    possrcs,
    posrecs,
    srctf,
    traces
) where {N}
       
    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_a = model.backend.Data.Array(possrcs)
    posrecs_a = model.backend.Data.Array(posrecs)
    srctf_a = model.backend.Data.Array(srctf)
    traces_a = model.backend.Data.Array(traces)

    ## ONLY 2D for now!!!
    Mxx = model.backend.Data.Array(momtens.Mxx)
    Mzz = model.backend.Data.Array(momtens.Mzz)
    Mxz = model.backend.Data.Array(momtens.Mxz)


    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        model.backend.forward_onestep_CPML!(model.velpartic..., model.stress...,
                                            model.λ_ihalf
                                            model.ρ, model.ρ_ihalf_jhalf,
                                            model.μ, model.μ_ihalf, model.μ_jhalf,
                                            model.dt, model.gridspacing...,
                                            model.ψ...,
                                            model.a_coeffs...,
                                            model.b_coeffs...,
                                            Mxx,Mzz,Mxz,
                                            possrcs_a, srctf_a, posrecs_a, traces_a, it,
                                            freetop, save_trace)
        
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
