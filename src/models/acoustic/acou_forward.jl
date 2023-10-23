swforward_1shot!(model::AcousticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimul{N},
    possrcs,
    posrecs,
    srctf,
    recs
) where {N}
    # Pressure arrays
    pold = model.pold
    pcur = model.pcur
    pnew = model.pnew
    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_a = model.backend.Data.Array(possrcs)
    posrecs_a = model.backend.Data.Array(posrecs)
    srctf_a = model.backend.Data.Array(srctf)
    traces_a = model.backend.Data.Array(recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        pold, pcur, pnew = model.backend.forward_onestep_CPML!(
            pold, pcur, pnew, model.fact,
            model.gridspacing..., model.halo,
            model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )
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
            copyto!(model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(recs.seismograms, traces_a)
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimul{N},
    possrcs,
    posrecs,
    srctf,
    recs
) where {N}
    # Pressure and velocity arrays
    pcur = model.pcur
    vcur = model.vcur
    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_a = model.backend.Data.Array(possrcs)
    posrecs_a = model.backend.Data.Array(posrecs)
    srctf_a = model.backend.Data.Array(srctf)
    traces_a = model.backend.Data.Array(recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        model.backend.forward_onestep_CPML!(
            pcur, vcur..., model.fact_m0, model.fact_m1_stag...,
            model.gridspacing..., model.halo,
            model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )
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
            copyto!(model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(recs.seismograms, traces_a)
end
