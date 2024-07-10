
swforward_1shot!(model::ElasticWaveSimulation, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::MomentTensorShot{T, 2, MomentTensor2D{T}}
) where {T}

    # scale source time function, etc.
    # find nearest grid points indexes for both sources and receivers
    # possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    # posrecs = find_nearest_grid_points(model, shot.recs.positions)

    # interpolation coefficients for sources
    srccoeij, srccoeval = spreadsrcrecinterp2D(model.grid.spacing, model.grid.size,
        shot.srcs.positions;
        nptssinc=4, xstart=0.0, zstart=0.0)
    # interpolation coefficients for receivers
    reccoeij, reccoeval = spreadsrcrecinterp2D(model.grid.spacing, model.grid.size,
        shot.recs.positions;
        nptssinc=4, xstart=0.0, zstart=0.0)

    # source time function
    srctf = shot.srcs.tf
    # moment tensors
    momtens = shot.srcs.momtens

    # Get computational grid and backend
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt

    # Wrap sources and receivers arrays
    srccoeij_bk = backend.Data.Array(srccoeij)
    srccoeval_bk = backend.Data.Array(srccoeval)
    reccoeij_bk = backend.Data.Array(reccoeij)
    reccoeval_bk = backend.Data.Array(reccoeval)
    
    srctf_bk = backend.Data.Array(srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)

    ## ONLY 2D for now!!!
    nsrcs = length(momtens)
    Mxx_bk = backend.Data.Array([momtens[i].Mxx for i in 1:nsrcs])
    Mzz_bk = backend.Data.Array([momtens[i].Mzz for i in 1:nsrcs])
    Mxz_bk = backend.Data.Array([momtens[i].Mxz for i in 1:nsrcs])

    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        backend.forward_onestep_CPML!(model,
                srccoeij_bk,
                srccoeval_bk,
                reccoeij_bk,
                reccoeval_bk,
                srctf_bk,
                traces_bk,
                it,
                Mxx_bk, Mzz_bk, Mxz_bk;
                save_trace=true)

        # Print timestep info
        if it % model.infoevery == 0
            # Move the cursor to the beginning to overwrite last line
            # ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
            # REPL.Terminals.clear_line(ter)
            # REPL.Terminals.cmove_line_up(ter)
            @info @sprintf(
                "Iteration: %d/%d, simulation time: %g s",
                it, nt,
                model.dt * (it - 1)
            )
        end

        # Save snapshot
        if snapenabled(model)
            savesnapshot!(model.snapshotter, "v" => model.grid.fields["v"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
    return
end

function spreadsrcrecinterp2D(
    gridspacing::NTuple{N, T},
    gridsize::NTuple{N, Int},
    positions::Matrix{T};
    nptssinc::Int=4, xstart::T=0.0, zstart::T=0.0
) where {T, N}
    nloc = size(positions, 1)
    Ndim = size(positions, 2)
    @assert Ndim == 2
    @assert N == Ndim

    Δx = gridspacing[1]
    Δz = gridspacing[2]
    nx, nz = gridsize[1:2]

    maxnumcoeff = nloc * (2 * nptssinc + 1)^Ndim
    coeij_tmp = zeros(Int, maxnumcoeff, Ndim + 1)
    coeval_tmp = zeros(T, maxnumcoeff)
    l = 0
    for p in 1:nloc
        # extract x and z position for source or receiver p
        xpos, zpos = positions[p, :]
        # compute grid indices and values of sinc coefficients
        xidx, zidx, xzcoeff = coeffsinc2D(xstart, zstart, Δx, Δz, xpos, zpos,
            nx, nz, [:monopole, :monopole]; npts=nptssinc)
        for j in eachindex(zidx)
            for i in eachindex(xidx)
                l += 1
                # id, i, j indices
                coeij_tmp[l, :] .= (p, xidx[i], zidx[j])
                # coefficients value
                coeval_tmp[l] = xzcoeff[i, j]
            end
        end
    end
    # keep only the valid part of the arrays
    coeij = coeij_tmp[1:l, :]
    coeval = coeval_tmp[1:l]

    return coeij, coeval
end
