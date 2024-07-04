
# Generic function
swforward_1shot!(model::AcousticWaveSimul, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

# Scaling for AcousticCDWaveSimul
@views function possrcrec_scaletf(model::AcousticWaveSimul, shot::Shot)
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    posrecs = find_nearest_grid_points(model, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(model.grid.gridspacing) .* (model.dt^2)
    # scale with velocity squared at each source position
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= model.matprop.vp[possrcs[s, :]...] .^ 2
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDCPMLWaveSimul{T, N},
    shot::Shot
) where {T, N}

    # Scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    # Get computational grid and backend
    grid = model.grid
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_bk = backend.Data.Array(possrcs)
    posrecs_bk = backend.Data.Array(posrecs)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                model.dt * (it - 1),
                maximum(abs.(Array(pcur)))
            )
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @info @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end

#####################################################

# Scaling for AcousticVDStaggeredCPMLWaveSimul
@views function possrcrec_scaletf(model::AcousticVDStaggeredCPMLWaveSimul, shot::Shot)
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    posrecs = find_nearest_grid_points(model, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(model.grid.gridspacing) .* (model.dt)
    # scale with velocity squared times density at each source position (its like dividing by m0)
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= model.matprop.vp[possrcs[s, :]...] .^ 2 .* model.matprop.rho[possrcs[s, :]...]
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimul{T, N},
    shot::Shot
) where {T, N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    # Get computational grid and backend
    grid = model.grid
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_bk = backend.Data.Array(possrcs)
    posrecs_bk = backend.Data.Array(posrecs)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf(
                "Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]",
                it,
                model.dt * (it - 1),
                maximum(abs.(Array(grid.fields["pcur"].value)))
            )
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @info @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(grid.fields["pcur"].value))))
            copyto!(model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)], grid.fields["pcur"].value)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end
