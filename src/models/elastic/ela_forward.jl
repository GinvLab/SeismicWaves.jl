
swforward_1shot!(model::ElasticWaveSimulation, args...) = swforward_1shot!(BoundaryConditionTrait(model), model, args...)

function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::MomentTensorShot{T, 2, MomentTensor2D{T}}
) where {T}
    # scale source time function
    srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz,
    reccoeij_ux, reccoeval_ux, reccoeij_uz, reccoeval_uz,
    scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)
    # moment tensors
    momtens = shot.srcs.momtens

    # Get computational grid and backend
    backend = select_backend(typeof(model), model.runparams.parall)

    # Numerics
    nt = model.nt

    # Wrap sources and receivers arrays
    nsrcs = size(shot.srcs.positions, 1)
    nrecs = size(shot.recs.positions, 1)
    srccoeij_xx  = [backend.Data.Array( srccoeij_xx[i]) for i in 1:nsrcs]
    srccoeval_xx = [backend.Data.Array(srccoeval_xx[i]) for i in 1:nsrcs]
    srccoeij_xz  = [backend.Data.Array( srccoeij_xz[i]) for i in 1:nsrcs]
    srccoeval_xz = [backend.Data.Array(srccoeval_xz[i]) for i in 1:nsrcs]
    reccoeij_ux  = [backend.Data.Array( reccoeij_ux[i]) for i in 1:nrecs]
    reccoeval_ux = [backend.Data.Array(reccoeval_ux[i]) for i in 1:nrecs]
    reccoeij_uz  = [backend.Data.Array( reccoeij_uz[i]) for i in 1:nrecs]
    reccoeval_uz = [backend.Data.Array(reccoeval_uz[i]) for i in 1:nrecs]
    srctf_bk = backend.Data.Array(scal_srctf)
    reduced_buf = [backend.zeros(T, nrecs), backend.zeros(T, nrecs)]
    traces_ux_bk_buf = [backend.zeros(T, size(reccoeij_ux[i], 1)) for i in 1:nrecs]
    traces_uz_bk_buf = [backend.zeros(T, size(reccoeij_uz[i], 1)) for i in 1:nrecs]
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

    ## ONLY 2D for now!!!
    nsrcs = length(momtens)
    Mxx_bk = backend.Data.Array([momtens[i].Mxx for i in 1:nsrcs])
    Mzz_bk = backend.Data.Array([momtens[i].Mzz for i in 1:nsrcs])
    Mxz_bk = backend.Data.Array([momtens[i].Mxz for i in 1:nsrcs])

    # Reset wavesim
    reset!(model)

    # Setup the print output 
    ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

    # Time loop
    for it in 1:nt
        backend.forward_onestep_CPML!(model,
            srccoeij_xx,
            srccoeval_xx,
            srccoeij_xz,
            srccoeval_xz,
            reccoeij_ux,
            reccoeval_ux,
            reccoeij_uz,
            reccoeval_uz,
            srctf_bk,
            reduced_buf,
            traces_ux_bk_buf,
            traces_uz_bk_buf,
            traces_bk,
            it,
            Mxx_bk, Mzz_bk, Mxz_bk;
            save_trace=true)

        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:forw)

        # Save snapshot
        if snapenabled(model)
            savesnapshot!(model.snapshotter, "ucur" => model.grid.fields["ucur"], it)
            savesnapshot!(model.snapshotter, "σ" => model.grid.fields["σ"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
    return
end

function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::ExternalForceShot{T, 2}
) where {T}
    # scale source time function
    srccoeij_ux, srccoeval_ux, srccoeij_uz, srccoeval_uz,
    reccoeij_ux, reccoeval_ux, reccoeij_uz, reccoeval_uz,
    scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

    # Get computational grid and backend
    backend = select_backend(typeof(model), model.runparams.parall)

    # Numerics
    nt = model.nt

    # Wrap sources and receivers arrays
    nsrcs = size(shot.srcs.positions, 1)
    nrecs = size(shot.recs.positions, 1)
    srccoeij_ux  = [backend.Data.Array( srccoeij_ux[i]) for i in 1:nsrcs]
    srccoeval_ux = [backend.Data.Array(srccoeval_ux[i]) for i in 1:nsrcs]
    srccoeij_uz  = [backend.Data.Array( srccoeij_uz[i]) for i in 1:nsrcs]
    srccoeval_uz = [backend.Data.Array(srccoeval_uz[i]) for i in 1:nsrcs]
    reccoeij_ux  = [backend.Data.Array( reccoeij_ux[i]) for i in 1:nrecs]
    reccoeval_ux = [backend.Data.Array(reccoeval_ux[i]) for i in 1:nrecs]
    reccoeij_uz  = [backend.Data.Array( reccoeij_uz[i]) for i in 1:nrecs]
    reccoeval_uz = [backend.Data.Array(reccoeval_uz[i]) for i in 1:nrecs]
    srctf_bk = backend.Data.Array(scal_srctf)
    reduced_buf = [backend.zeros(T, nrecs), backend.zeros(T, nrecs)]
    traces_ux_bk_buf = [backend.zeros(T, size(reccoeij_ux[i], 1)) for i in 1:nrecs]
    traces_uz_bk_buf = [backend.zeros(T, size(reccoeij_uz[i], 1)) for i in 1:nrecs]
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

    # Reset wavesim
    reset!(model)

    # Setup the print output 
    ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

    # Time loop
    for it in 1:nt
        backend.forward_onestep_CPML!(
            model,
            srccoeij_ux,
            srccoeval_ux,
            srccoeij_uz,
            srccoeval_uz,
            reccoeij_ux,
            reccoeval_ux,
            reccoeij_uz,
            reccoeval_uz,
            srctf_bk,
            reduced_buf,
            traces_ux_bk_buf,
            traces_uz_bk_buf,
            traces_bk,
            it;
            save_trace=true
        )

        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:forw)

        # Save snapshot
        if snapenabled(model)
            savesnapshot!(model.snapshotter, "ucur" => model.grid.fields["ucur"], it)
            savesnapshot!(model.snapshotter, "σ" => model.grid.fields["σ"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
    return
end

function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::PSDMomentTensorShot{T, 2}
) where {T}
    # For each reference receiver
    for (ii, irecref) in enumerate(shot.recs.refs)
        # For each component (2D so x and z)
        for d in shot.recs.comps
            # Get computational grid and backend
            backend = select_backend(typeof(model), model.runparams.parall)

            # Scale source time function
            srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz,
            reccoeij_ux, reccoeval_ux, reccoeij_uz, reccoeval_uz,
            scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

            # Numerics
            nt = model.nt

            # Wrap sources and receivers arrays
            nsrcs = size(shot.srcs.positions, 1)
            nrecs = size(shot.recs.positions, 1)
            srccoeij_xx  = [backend.Data.Array( srccoeij_xx[i]) for i in 1:nsrcs]
            srccoeval_xx = [backend.Data.Array(srccoeval_xx[i]) for i in 1:nsrcs]
            srccoeij_xz  = [backend.Data.Array( srccoeij_xz[i]) for i in 1:nsrcs]
            srccoeval_xz = [backend.Data.Array(srccoeval_xz[i]) for i in 1:nsrcs]
            reccoeij_ux  = [backend.Data.Array( reccoeij_ux[i]) for i in 1:nrecs]
            reccoeval_ux = [backend.Data.Array(reccoeval_ux[i]) for i in 1:nrecs]
            reccoeij_uz  = [backend.Data.Array( reccoeij_uz[i]) for i in 1:nrecs]
            reccoeval_uz = [backend.Data.Array(reccoeval_uz[i]) for i in 1:nrecs]
            refreccoeij_ux  = backend.Data.Array( reccoeij_ux[irecref])
            refreccoeval_ux = backend.Data.Array(reccoeval_ux[irecref])
            refreccoeij_uz  = backend.Data.Array( reccoeij_uz[irecref])
            refreccoeval_uz = backend.Data.Array(reccoeval_uz[irecref])
            refrecscal_stf  = backend.Data.Array(scal_srctf)

            # Wrap traces arrays
            reduced_buf_stress = [backend.zeros(T, nsrcs), backend.zeros(T, nsrcs), backend.zeros(T, nsrcs)]
            traces_xx_bk_buf = [backend.zeros(T, size(srccoeij_xx[i], 1)) for i in 1:nsrcs]
            traces_xz_bk_buf = [backend.zeros(T, size(srccoeij_xz[i], 1)) for i in 1:nsrcs]
            traces_zz_bk_buf = [backend.zeros(T, size(srccoeij_xx[i], 1)) for i in 1:nsrcs]
            traces_xx_bk = backend.zeros(T, nt, nsrcs)
            traces_xz_bk = backend.zeros(T, nt, nsrcs)
            traces_zz_bk = backend.zeros(T, nt, nsrcs)
            stf_xx_bk = backend.zeros(T, nt, nsrcs)
            stf_zz_bk = backend.zeros(T, nt, nsrcs)
            stf_xz_bk = backend.zeros(T, nt, nsrcs)
            reduced_buf_displ = [backend.zeros(T, nrecs), backend.zeros(T, nrecs)]
            traces_ux_bk_buf = [backend.zeros(T, size(reccoeij_ux[i], 1)) for i in 1:nrecs]
            traces_uz_bk_buf = [backend.zeros(T, size(reccoeij_uz[i], 1)) for i in 1:nrecs]
            traces_bk = backend.zeros(T, nt*2 + 1, 2, nrecs)

            # Reset wavesim
            reset!(model)

            # Setup the print output 
            ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

            @info "Computing CCs for component #$d of reference receiver #$irecref"

            # Out-propagation time loop
            for it in 1:nt
                backend.forward_onestep_ccout_CPML!(
                    model,
                    d == 1 ? refreccoeij_ux : refreccoeij_uz,
                    d == 1 ? refreccoeval_ux : refreccoeval_uz,
                    srccoeij_xx,
                    srccoeval_xx,
                    srccoeij_xz,
                    srccoeval_xz,
                    refrecscal_stf,
                    reduced_buf_stress,
                    traces_xx_bk_buf,
                    traces_zz_bk_buf,
                    traces_xz_bk_buf,
                    traces_xx_bk,
                    traces_zz_bk,
                    traces_xz_bk,
                    it;
                    d=d,
                    save_trace=true
                )

                # Print timestep info
                printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:outprop)
            end

            # Scale PSD sources by moment tensor
            for s in 1:nsrcs
                stf_xx_bk[:, s] .= traces_xx_bk[:, s] .* shot.srcs.psd[s].Mxx
                stf_zz_bk[:, s] .= traces_zz_bk[:, s] .* shot.srcs.psd[s].Mzz
                stf_xz_bk[:, s] .= traces_xz_bk[:, s] .* shot.srcs.psd[s].Mxz
            end

            reset!(model)

            # In-propagation time loop
            for it in nt:-1:-nt
                backend.forward_onestep_ccin_CPML!(
                    model,
                    srccoeij_xx,
                    srccoeval_xx,
                    srccoeij_xz,
                    srccoeval_xz,
                    reccoeij_ux,
                    reccoeval_ux,
                    reccoeij_uz,
                    reccoeval_uz,
                    stf_xx_bk,
                    stf_zz_bk,
                    stf_xz_bk,
                    reduced_buf_displ,
                    traces_ux_bk_buf,
                    traces_uz_bk_buf,
                    traces_bk,
                    it, nt;
                    save_trace=true
                )

                # Print timestep info
                printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:inprop)
            end

            # Save cross-correlations
            copyto!(@view(shot.recs.crosscorrelations[:, d, :, ii, :]), traces_bk)
        end
    end
end

function swforward_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::PSDExternalForceShot{T, 2}
) where {T}
    # For each reference receiver
    for (ii, irecref) in enumerate(shot.recs.refs)
        # For each component (2D so x and z)
        for d in shot.recs.comps
            # Get computational grid and backend
            backend = select_backend(typeof(model), model.runparams.parall)

            # Scale source time function
            srccoeij_ux, srccoeval_ux, srccoeij_uz, srccoeval_uz,
            reccoeij_ux, reccoeval_ux, reccoeij_uz, reccoeval_uz,
            scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

            # Numerics
            nt = model.nt

            # Wrap sources and receivers arrays
            nsrcs = size(shot.srcs.positions, 1)
            nrecs = size(shot.recs.positions, 1)
            srccoeij_ux  = [backend.Data.Array( srccoeij_ux[i]) for i in 1:nsrcs]
            srccoeval_ux = [backend.Data.Array(srccoeval_ux[i]) for i in 1:nsrcs]
            srccoeij_uz  = [backend.Data.Array( srccoeij_uz[i]) for i in 1:nsrcs]
            srccoeval_uz = [backend.Data.Array(srccoeval_uz[i]) for i in 1:nsrcs]
            reccoeij_ux  = [backend.Data.Array( reccoeij_ux[i]) for i in 1:nrecs]
            reccoeval_ux = [backend.Data.Array(reccoeval_ux[i]) for i in 1:nrecs]
            reccoeij_uz  = [backend.Data.Array( reccoeij_uz[i]) for i in 1:nrecs]
            reccoeval_uz = [backend.Data.Array(reccoeval_uz[i]) for i in 1:nrecs]
            refreccoeij_ux  = backend.Data.Array( reccoeij_ux[irecref])
            refreccoeval_ux = backend.Data.Array(reccoeval_ux[irecref])
            refreccoeij_uz  = backend.Data.Array( reccoeij_uz[irecref])
            refreccoeval_uz = backend.Data.Array(reccoeval_uz[irecref])
            refrecscal_stf  = backend.Data.Array(scal_srctf)

            # Wrap traces arrays
            reduced_buf_displ_out = [backend.zeros(T, nsrcs), backend.zeros(T, nsrcs)]
            traces_ux_bk_buf_out = [backend.zeros(T, size(srccoeij_ux[i], 1)) for i in 1:nsrcs]
            traces_uz_bk_buf_out = [backend.zeros(T, size(srccoeij_uz[i], 1)) for i in 1:nsrcs]
            traces_ux_bk = backend.zeros(T, nt, nsrcs)
            traces_uz_bk = backend.zeros(T, nt, nsrcs)
            stf_ux_bk = backend.zeros(T, nt, nsrcs)
            stf_uz_bk = backend.zeros(T, nt, nsrcs)
            reduced_buf_displ_in = [backend.zeros(T, nrecs), backend.zeros(T, nrecs)]
            traces_ux_bk_buf_in = [backend.zeros(T, size(reccoeij_ux[i], 1)) for i in 1:nrecs]
            traces_uz_bk_buf_in = [backend.zeros(T, size(reccoeij_uz[i], 1)) for i in 1:nrecs]
            traces_bk = backend.zeros(T, nt*2 + 1, 2, nrecs)

            # Reset wavesim
            reset!(model)

            # Setup the print output 
            ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

            @info "Computing CCs for component #$d of reference receiver #$irecref"

            # Out-propagation time loop
            for it in 1:nt
                backend.forward_onestep_ccout_CPML!(
                    model,
                    d == 1 ? refreccoeij_ux : refreccoeij_uz,
                    d == 1 ? refreccoeval_ux : refreccoeval_uz,
                    srccoeij_ux,
                    srccoeval_ux,
                    srccoeij_uz,
                    srccoeval_uz,
                    refrecscal_stf,
                    reduced_buf_displ_out,
                    traces_ux_bk_buf_out,
                    traces_uz_bk_buf_out,
                    traces_ux_bk,
                    traces_uz_bk,
                    it;
                    d=d,
                    save_trace=true
                )

                # Print timestep info
                printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:outprop)
            end

            # Scale PSD sources
            for s in 1:nsrcs
                stf_ux_bk[:, s] .= (traces_ux_bk[:, s] .* shot.srcs.psd[s][1] .+
                                    traces_uz_bk[:, s] .* shot.srcs.psd[s][2]) .* shot.srcs.psd[s][1]
                stf_uz_bk[:, s] .= (traces_ux_bk[:, s] .* shot.srcs.psd[s][1] .+
                                    traces_uz_bk[:, s] .* shot.srcs.psd[s][2]) .* shot.srcs.psd[s][2]
            end

            reset!(model)

            # In-propagation time loop
            for it in nt:-1:-nt
                backend.forward_onestep_ccin_CPML!(
                    model,
                    srccoeij_ux,
                    srccoeval_ux,
                    srccoeij_uz,
                    srccoeval_uz,
                    reccoeij_ux,
                    reccoeval_ux,
                    reccoeij_uz,
                    reccoeval_uz,
                    stf_ux_bk,
                    stf_uz_bk,
                    reduced_buf_displ_in,
                    traces_ux_bk_buf_in,
                    traces_uz_bk_buf_in,
                    traces_bk,
                    it, nt;
                    save_trace=true
                )

                # Print timestep info
                printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:inprop)
            end

            # Save cross-correlations
            copyto!(@view(shot.recs.crosscorrelations[:, d, :, ii, :]), traces_bk)
        end
    end
end