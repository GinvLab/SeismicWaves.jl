
swforward_1shot!(model::ElasticWaveSimulation, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::MomentTensorShot{T, 2, MomentTensor2D{T}}
) where {T}
    # scale source time function
    srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz,
    reccoeij_vx, reccoeval_vx, reccoeij_vz, reccoeval_vz,
    scal_srctf = possrcrec_scaletf(model, shot)
    # moment tensors
    momtens = shot.srcs.momtens

    # Get computational grid and backend
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt

    # Wrap sources and receivers arrays
    srccoeij_xx = backend.Data.Array(srccoeij_xx)
    srccoeval_xx = backend.Data.Array(srccoeval_xx)
    srccoeij_xz = backend.Data.Array(srccoeij_xz)
    srccoeval_xz = backend.Data.Array(srccoeval_xz)
    reccoeij_vx = backend.Data.Array(reccoeij_vx)
    reccoeval_vx = backend.Data.Array(reccoeval_vx)
    reccoeij_vz = backend.Data.Array(reccoeij_vz)
    reccoeval_vz = backend.Data.Array(reccoeval_vz)

    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

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
            srccoeij_xx,
            srccoeval_xx,
            srccoeij_xz,
            srccoeval_xz,
            reccoeij_vx,
            reccoeval_vx,
            reccoeij_vz,
            reccoeval_vz,
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

@views function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::ExternalForceShot{T, 2}
) where {T}
    # scale source time function
    srccoeij_vx, srccoeval_vx,
    srccoeij_vz, srccoeval_vz,
    reccoeij_vx, reccoeval_vx,
    reccoeij_vz, reccoeval_vz,
    scal_srctf = possrcrec_scaletf(model, shot)

    # Get computational grid and backend
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt

    # Wrap sources and receivers arrays
    srccoeij_vx = backend.Data.Array(srccoeij_vx)
    srccoeval_vx = backend.Data.Array(srccoeval_vx)
    srccoeij_vz = backend.Data.Array(srccoeij_vz)
    srccoeval_vz = backend.Data.Array(srccoeval_vz)
    reccoeij_vx = backend.Data.Array(reccoeij_vx)
    reccoeval_vx = backend.Data.Array(reccoeval_vx)
    reccoeij_vz = backend.Data.Array(reccoeij_vz)
    reccoeval_vz = backend.Data.Array(reccoeval_vz)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

    # Reset wavesim
    reset!(model)

    # Time loop
    for it in 1:nt
        backend.forward_onestep_CPML!(
            model,
            srccoeij_vx,
            srccoeval_vx,
            srccoeij_vz,
            srccoeval_vz,
            reccoeij_vx,
            reccoeval_vx,
            reccoeij_vz,
            reccoeval_vz,
            srctf_bk,
            traces_bk,
            it;
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