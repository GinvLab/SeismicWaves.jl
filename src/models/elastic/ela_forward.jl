
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
    backend = select_backend(typeof(model), model.parall)

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
            savesnapshot!(model.snapshotter, "ucur" => model.grid.fields["ucur"], it)
            savesnapshot!(model.snapshotter, "σ" => model.grid.fields["σ"], it)
        end
    end

    # Save traces
    copyto!(shot.recs.seismograms, traces_bk)
    return
end

# function swforward_1shot!(
#     ::CPMLBoundaryCondition,
#     model::ElasticIsoCPMLWaveSimulation{T, 2},
#     shot::ExternalForceShot{T, 2}
# ) where {T}
#     # scale source time function
#     srccoeij_ux, srccoeval_ux,
#     srccoeij_uz, srccoeval_uz,
#     reccoeij_ux, reccoeval_ux,
#     reccoeij_uz, reccoeval_uz,
#     scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

#     # Get computational grid and backend
#     backend = select_backend(typeof(model), model.parall)

#     # Numerics
#     nt = model.nt

#     # Wrap sources and receivers arrays
#     nsrcs = size(shot.srcs.positions, 1)
#     nrecs = size(shot.recs.positions, 1)
#     srccoeij_ux  = [backend.Data.Array( srccoeij_ux[i]) for i in 1:nsrcs]
#     srccoeval_ux = [backend.Data.Array(srccoeval_ux[i]) for i in 1:nsrcs]
#     srccoeij_uz  = [backend.Data.Array( srccoeij_uz[i]) for i in 1:nsrcs]
#     srccoeval_uz = [backend.Data.Array(srccoeval_uz[i]) for i in 1:nsrcs]
#     reccoeij_ux  = [backend.Data.Array( reccoeij_ux[i]) for i in 1:nrecs]
#     reccoeval_ux = [backend.Data.Array(reccoeval_ux[i]) for i in 1:nrecs]
#     reccoeij_uz  = [backend.Data.Array( reccoeij_uz[i]) for i in 1:nrecs]
#     reccoeval_uz = [backend.Data.Array(reccoeval_uz[i]) for i in 1:nrecs]
#     srctf_bk = backend.Data.Array(scal_srctf)
#     reduced_buf = [backend.zeros(T, nrecs), backend.zeros(T, nrecs)]
#     traces_ux_bk_buf = [backend.zeros(T, size(reccoeij_ux[i], 1)) for i in 1:nrecs]
#     traces_uz_bk_buf = [backend.zeros(T, size(reccoeij_uz[i], 1)) for i in 1:nrecs]
#     traces_bk = backend.zeros(T, size(shot.recs.seismograms))

#     # Reset wavesim
#     reset!(model)

#     # Time loop
#     for it in 1:nt
#         backend.forward_onestep_CPML!(
#             model,
#             srccoeij_ux,
#             srccoeval_ux,
#             srccoeij_uz,
#             srccoeval_uz,
#             reccoeij_ux,
#             reccoeval_ux,
#             reccoeij_uz,
#             reccoeval_uz,
#             srctf_bk,
#             reduced_buf,
#             traces_ux_bk_buf,
#             traces_uz_bk_buf,
#             traces_bk,
#             it;
#             save_trace=true)

#         # Print timestep info
#         if it % model.infoevery == 0
#             # Move the cursor to the beginning to overwrite last line
#             # ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)
#             # REPL.Terminals.clear_line(ter)
#             # REPL.Terminals.cmove_line_up(ter)
#             @info @sprintf(
#                 "Iteration: %d/%d, simulation time: %g s",
#                 it, nt,
#                 model.dt * (it - 1)
#             )
#         end

#         # Save snapshot
#         if snapenabled(model)
#             savesnapshot!(model.snapshotter, "ucur" => model.grid.fields["ucur"], it)
#             savesnapshot!(model.snapshotter, "σ" => model.grid.fields["σ"], it)
#         end
#     end

#     # Save traces
#     copyto!(shot.recs.seismograms, traces_bk)
#     return
# end
