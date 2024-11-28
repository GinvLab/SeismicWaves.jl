
# Generic function
swforward_1shot!(model::AcousticWaveSimulation, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

# Scaling for AcousticCDWaveSimulation
@views function possrcrec_scaletf(model::AcousticWaveSimulation{T}, shot::ScalarShot{T}) where {T}
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    posrecs = find_nearest_grid_points(model, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(model.grid.spacing) .* (model.dt^2)
    # scale with velocity squared at each source position
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= model.matprop.vp[possrcs[s, :]...] .^ 2
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDCPMLWaveSimulation{T, N},
    shot::ScalarShot{T}
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
        backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Iteration: %d, simulation time: %g [s]", it, model.dt * (it - 1))
        end

        # Save snapshot
        if snapenabled(model)
            savesnapshot!(model.snapshotter, "pcur" => grid.fields["pcur"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end

#####################################################

# Scaling for AcousticVDStaggeredCPMLWaveSimulation
@views function possrcrec_scaletf(model::AcousticVDStaggeredCPMLWaveSimulation{T}, shot::ScalarShot{T}) where {T}
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    posrecs = find_nearest_grid_points(model, shot.recs.positions)

    # source time function 
    # scale with boxcar and timestep size
    scal_srctf = shot.srcs.tf ./ prod(model.grid.spacing) .* (model.dt)
    # scale with velocity squared times density at each source position (its like dividing by m0)
    for s in axes(scal_srctf, 2)
        scal_srctf[:, s] .*= model.matprop.vp[possrcs[s, :]...] .^ 2 .* model.matprop.rho[possrcs[s, :]...]
    end

    return possrcs, posrecs, scal_srctf
end

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimulation{T, N},
    shot::ScalarShot{T}
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
        backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Iteration: %d, simulation time: %g [s]", it, model.dt * (it - 1))
        end

        # Save snapshot
        if snapenabled(model)
            savesnapshot!(model.snapshotter, "pcur" => grid.fields["pcur"], it)
            savesnapshot!(model.snapshotter, "vcur" => grid.fields["vcur"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
end
