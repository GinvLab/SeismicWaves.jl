
# Generic function
swforward_1shot!(wavsim::AcousticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(wavsim), wavsim, args...)

# Scaling for AcousticCDWaveSimul
@views function possrcrec_scaletf(wavsim::AcousticWaveSimul, shot::Shot)
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
    wavsim::AcousticCDCPMLWaveSimul{T, N},
    shot::Shot
) where {T, N}

    # Scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(wavsim, shot)

    # Get computational grid and backend
    grid = wavsim.grid
    backend = select_backend(typeof(wavsim), wavsim.parall)

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
        backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
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
@views function possrcrec_scaletf(wavsim::AcousticVDStaggeredCPMLWaveSimul, shot::Shot)
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(wavsim, shot.srcs.positions)
    posrecs = find_nearest_grid_points(wavsim, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(wavsim.grid.gridspacing) .* (wavsim.dt)
    # scale with velocity squared times density at each source position (its like dividing by m0)
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= wavsim.matprop.vp[possrcs[s, :]...] .^ 2 .* wavsim.matprop.rho[possrcs[s, :]...]
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    wavsim::AcousticVDStaggeredCPMLWaveSimul{T, N},
    shot::Shot
) where {T, N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(wavsim, shot)

    # Get computational grid and backend
    grid = wavsim.grid
    backend = select_backend(typeof(wavsim), wavsim.parall)

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
        backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % wavsim.infoevery == 0
            @info @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                wavsim.dt * (it - 1),
                maximum(abs.(Array(grid.fields["pcur"].value)))
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            @info @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(grid.fields["pcur"].value))))
            copyto!(wavsim.snapshots[fill(Colon(), N)..., div(it, wavsim.snapevery)], grid.fields["pcur"].value)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end
