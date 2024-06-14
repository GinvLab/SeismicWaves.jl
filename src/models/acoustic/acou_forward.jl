
# Generic function
swforward_1shot!(wavsim::AcousticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(wavsim), wavsim, args...)

# Scaling for AcousticCDWaveSimul
@views function possrcrec_scaletf(wavsim::AcousticCDCPMLWaveSimul{N},
    shot::Shot) where {N}
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(wavsim, shot.srcs.positions)
    posrecs = find_nearest_grid_points(wavsim, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(wavsim.grid.gridspacing) .* (wavsim.dt^2)
    # scale with velocity squared at each source position
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= wavsim.matprop.vp[possrcs[s, :]...] .^ 2
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    wavsim::AcousticCDCPMLWaveSimul{N},
    shot::Shot
) where {N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(wavsim, shot)

    grid = wavsim.grid
    backend = select_backend(typeof(wavsim), wavsim.parall)

    # Pressure arrays
    pold = grid.fields["pold"].value
    pcur = grid.fields["pcur"].value
    pnew = grid.fields["pnew"].value
    # Numerics
    nt = wavsim.nt
    # Wrap sources and receivers arrays
    possrcs_bk = backend.Data.Array(possrcs)
    posrecs_bk = backend.Data.Array(posrecs)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(wavsim)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, grid.fields["fact"].value,
            wavsim.grid.gridspacing..., wavsim.halo,
            grid.fields["ψ"].value..., grid.fields["ξ"].value..., grid.fields["a_pml"].value..., grid.fields["b_pml"].value...,
            possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it
        )
        # Print timestep info
        if it % wavsim.infoevery == 0
            @info @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                wavsim.dt * (it - 1),
                maximum(abs.(Array(pcur)))
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            @info @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(wavsim.snapshots[fill(Colon(), N)..., div(it, wavsim.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end

#####################################################

# Scaling for AcousticVDStaggeredCPMLWaveSimul
@views function possrcrec_scaletf(wavsim::AcousticVDStaggeredCPMLWaveSimul{N},
    shot::Shot) where {N}
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(wavsim, shot.srcs.positions)
    posrecs = find_nearest_grid_points(wavsim, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(wavsim.gridspacing) .* (wavsim.dt)
    # scale with velocity squared times density at each source position (its like dividing by m0)
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= wavsim.matprop.vp[possrcs[s, :]...] .^ 2 .* wavsim.matprop.rho[possrcs[s, :]...]
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    wavsim::AcousticVDStaggeredCPMLWaveSimul{N},
    shot::Shot
) where {N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(wavsim, shot)

    # Pressure and velocity arrays
    pcur = wavsim.pcur
    vcur = wavsim.vcur
    # Numerics
    nt = wavsim.nt
    # Wrap sources and receivers arrays
    possrcs_bk = wavsim.backend.Data.Array(possrcs)
    posrecs_bk = wavsim.backend.Data.Array(posrecs)
    srctf_bk = wavsim.backend.Data.Array(scal_srctf)
    traces_bk = wavsim.backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(wavsim)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        wavsim.backend.forward_onestep_CPML!(
            pcur, vcur..., wavsim.fact_m0, wavsim.fact_m1_stag...,
            wavsim.gridspacing..., wavsim.halo,
            wavsim.ψ..., wavsim.ξ..., wavsim.a_coeffs..., wavsim.b_coeffs...,
            possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it
        )
        # Print timestep info
        if it % wavsim.infoevery == 0
            @info @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                wavsim.dt * (it - 1),
                maximum(abs.(Array(pcur)))
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            @info @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(wavsim.snapshots[fill(Colon(), N)..., div(it, wavsim.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end
