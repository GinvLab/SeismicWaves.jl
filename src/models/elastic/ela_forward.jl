
swforward_1shot!(wavsim::ElasticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(wavsim), wavsim, args...)


@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    wavsim::ElasticIsoCPMLWaveSimul{N},
    shot::Shot
    # possrcs::Matrix{<:Integer},
    # posrecs::Matrix{<:Integer},
    # srctf,
    # recs
    ) where {N}

    @show propertynames(wavsim,true)

    # scale source time function, etc.
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(wavsim, shot.srcs.positions)
    posrecs = find_nearest_grid_points(wavsim, shot.recs.positions)

    srctf = shot.srcs.tf
    momtens = shot.srcs.momtens

    # Numerics
    nt = wavsim.nt
    # Wrap sources and receivers arrays
    possrcs_bk = wavsim.backend.Data.Array(possrcs)
    posrecs_bk = wavsim.backend.Data.Array(posrecs)
    srctf_bk  = wavsim.backend.Data.Array(srctf)
    traces_bk = wavsim.backend.Data.Array(shot.recs.seismograms)

    ## ONLY 2D for now!!!
    Mxx = wavsim.backend.Data.Array(momtens.Mxx)
    Mzz = wavsim.backend.Data.Array(momtens.Mzz)
    Mxz = wavsim.backend.Data.Array(momtens.Mxz)

    # Reset wavesim
    reset!(wavsim)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        wavsim.backend.forward_onestep_CPML!(wavsim,
                                             possrcs_bk,
                                             srctf_bk,
                                             posrecs_bk,
                                             traces_bk,
                                             it,
                                             Mxx,Mzz,Mxz,
                                             save_trace=true)

                                            # wavsim.velpartic..., wavsim.stress...,
                                            # wavsim.λ_ihalf
                                            # wavsim.ρ, wavsim.ρ_ihalf_jhalf,
                                            # wavsim.μ, wavsim.μ_ihalf, wavsim.μ_jhalf,
                                            # wavsim.dt, wavsim.gridspacing...,
                                            # wavsim.ψ...,
                                            # wavsim.a_coeffs...,
                                            # wavsim.b_coeffs...,
                                            # Mxx,Mzz,Mxz,
                                            # possrcs_a, srctf_a, posrecs_a, traces_a, it,
                                            # freetop, save_trace)
        
        # Print timestep info
        if it % wavsim.infoevery == 0
            # # Move the cursor to the beginning to overwrite last line
            # ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
            # REPL.Terminals.clear_line(ter)
            # REPL.Terminals.cmove_line_up(ter)
            @debug @sprintf(
                "Iteration: %d, simulation time: %g [s]",
                it,
                wavsim.dt * (it - 1)
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            error("Snapshot for elastic not yet implemented...")
            #@debug @sprintf("Snapping iteration: %d", it)
            #wavsim.snapshots[fill(Colon(), N)..., div(it, wavsim.snapevery)] .= Array(pcur)
        end
    end

    # Save traces
    copyto!(recs.seismograms, traces_bk)
    return
end
