
swforward_1shot!(wavsim::ElasticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(wavsim), wavsim, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    wavsim::ElasticIsoWaveSimul{N},
    possrcs::Matrix{<:Integer},
    posrecs::Matrix{<:Integer},
    srctf,
    recs
) where {N}
       
    # Numerics
    nt = wavsim.nt
    # Wrap sources and receivers arrays
    possrcs_a = wavsim.backend.Data.Array(possrcs)
    posrecs_a = wavsim.backend.Data.Array(posrecs)
    srctf_a  = wavsim.backend.Data.Array(srctf)
    traces_a = wavsim.backend.Data.Array(recs.seismograms)

    ## ONLY 2D for now!!!
    Mxx = wavsim.backend.Data.Array(momtens.Mxx)
    Mzz = wavsim.backend.Data.Array(momtens.Mzz)
    Mxz = wavsim.backend.Data.Array(momtens.Mxz)

    # Reset wavesim
    reset!(wavsim)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        wavsim.backend.forward_onestep_CPML!(wavsim, possrcs_a, srctf_a, posrecs_a, traces_a, it,
                                            freetop, save_trace)

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
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            wavsim.snapshots[fill(Colon(), N)..., div(it, wavsim.snapevery)] .= Array(pcur)
        end
    end

    # Save traces
    copyto!(recs.seismograms, traces_a)
    return
end
