
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

    # scale source time function, etc.
    # find nearest grid points indexes for both sources and receivers
    # possrcs = find_nearest_grid_points(wavsim, shot.srcs.positions)
    # posrecs = find_nearest_grid_points(wavsim, shot.recs.positions)

    if N == 2
        # interpolation coefficients for sources
        srccoeij, srccoeval = spreadsrcrecinterp2D(wavsim.gridspacing, wavsim.gridsize,
            shot.srcs.positions;
            nptssinc=4, xstart=0.0, zstart=0.0)
        # interpolation coefficients for receivers
        reccoeij, reccoeval = spreadsrcrecinterp2D(wavsim.gridspacing, wavsim.gridsize,
            shot.recs.positions;
            nptssinc=4, xstart=0.0, zstart=0.0)
    elseif N == 3
        error("swforward_1shot!(): Elastic 3D not yet implemented!")
    end

    # source time function
    srctf = shot.srcs.tf
    # moment tensors
    momtens = shot.srcs.momtens

    # Numerics
    nt = wavsim.nt

    # Wrap sources and receivers arrays

    # possrcs_bk = wavsim.backend.Data.Array(possrcs)
    # posrecs_bk = wavsim.backend.Data.Array(posrecs)

    srccoeij_bk = wavsim.backend.Data.Array(srccoeij)
    srccoeval_bk = wavsim.backend.Data.Array(srccoeval)
    reccoeij_bk = wavsim.backend.Data.Array(reccoeij)
    reccoeval_bk = wavsim.backend.Data.Array(reccoeval)

    srctf_bk = wavsim.backend.Data.Array(srctf)
    traces_bk = wavsim.backend.Data.Array(shot.recs.seismograms)

    if N == 2
        ## ONLY 2D for now!!!
        Mxx_bk = wavsim.backend.Data.Array(momtens.Mxx)
        Mzz_bk = wavsim.backend.Data.Array(momtens.Mzz)
        Mxz_bk = wavsim.backend.Data.Array(momtens.Mxz)
    elseif N == 3
        error("swforward_1shot!(): Elastic 3D not yet implemented!")
    end

    # Reset wavesim
    reset!(wavsim)
    # Zeros traces (there is a summation in record_receivers2D!()
    traces_bk .= 0.0

    # Time loop
    for it in 1:nt
        if N == 2
            # Compute one forward step
            wavsim.backend.forward_onestep_CPML!(wavsim,
                srccoeij_bk,
                srccoeval_bk,
                reccoeij_bk,
                reccoeval_bk,
                srctf_bk,
                traces_bk,
                it,
                Mxx_bk, Mzz_bk, Mxz_bk;
                save_trace=true)
            # # Compute one forward step
            # wavsim.backend.forward_onestep_CPML!(wavsim,
            #                                      possrcs_bk,
            #                                      srctf_bk,
            #                                      posrecs_bk,
            #                                      traces_bk,
            #                                      it,
            #                                      Mxx_bk,Mzz_bk,Mxz_bk,
            #                                      save_trace=true)
        else
            error("swforward_1shot!(): Elastic 3D not yet implemented!")
        end

        # Print timestep info
        if it % wavsim.infoevery == 0
            # Move the cursor to the beginning to overwrite last line
            # ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
            # REPL.Terminals.clear_line(ter)
            # REPL.Terminals.cmove_line_up(ter)
            @info @sprintf(
                "Iteration: %d/%d, simulation time: %g s",
                it, nt,
                wavsim.dt * (it - 1)
            )
        end

        # Save snapshot
        if snapenabled(wavsim) && it % wavsim.snapevery == 0
            #error("Snapshot for elastic not yet implemented...")
            @info @sprintf("Snapping iteration: %d", it)
            dummyidxs = fill(Colon(), N)
            # Vx
            wavsim.snapshots[1][dummyidxs..., div(it, wavsim.snapevery)] .= Array(wavsim.velpartic.vx)
            # Vz
            wavsim.snapshots[2][dummyidxs..., div(it, wavsim.snapevery)] .= Array(wavsim.velpartic.vz)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
    return
end

function spreadsrcrecinterp2D(gridspacing::NTuple{N, Real}, gridsize::NTuple{N, Integer},
    positions::Matrix{<:Real};
    nptssinc::Int=4, xstart::Real=0.0, zstart::Real=0.0) where {N}
    nloc = size(positions, 1)
    Ndim = size(positions, 2)
    @assert Ndim == 2
    @assert N == Ndim

    Δx = gridspacing[1]
    Δz = gridspacing[2]
    nx, nz = gridsize[1:2]

    maxnumcoeff = nloc * (2 * nptssinc + 1)^Ndim
    coeij_tmp = zeros(Int, maxnumcoeff, Ndim + 1)
    coeval_tmp = zeros(maxnumcoeff)
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
