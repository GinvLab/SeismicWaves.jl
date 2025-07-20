swgradient_1shot!(model::ElasticWaveSimulation, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::MomentTensorShot{T, 2, MomentTensor2D{T}},
    misfitshot::AbstractMisfit
)::Dict{String, Array{T, 2}} where {T}
    # scale source time function
    srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz,
    reccoeij_ux, reccoeval_ux, reccoeij_uz, reccoeval_uz,
    scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)
    # moment tensors
    momtens = shot.srcs.momtens

    # Get computational grid, checkpointer and backend
    grid = model.grid
    checkpointer = model.checkpointer
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

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(
            model,
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
            save_trace=true
        )

        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjforw)

        # Save checkpoint
        savecheckpoint!(checkpointer, "ucur" => grid.fields["ucur"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂x" => grid.fields["ψ_∂σ∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂z" => grid.fields["ψ_∂σ∂z"], it)
        savecheckpoint!(checkpointer, "ψ_∂u∂x" => grid.fields["ψ_∂u∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂u∂z" => grid.fields["ψ_∂u∂z"], it)
    end

    @debug "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    adjoint_src = backend.Data.Array(.-∂χ_∂u(misfitshot, shot.recs))

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(
            model,
            reccoeij_ux,
            reccoeval_ux,
            reccoeij_uz,
            reccoeval_uz,
            adjoint_src,
            reduced_buf,
            it
        )
        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjback)

        # Check if out of save buffer
        if !issaved(checkpointer, "ucur", it - 2)
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            initrecover!(checkpointer)
            copyto!(grid.fields["uold"], getsaved(checkpointer, "ucur", checkpointer.curr_checkpoint - 1))
            copyto!(grid.fields["ucur"], getsaved(checkpointer, "ucur", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂x"], getsaved(checkpointer, "ψ_∂σ∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂z"], getsaved(checkpointer, "ψ_∂σ∂z", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂u∂x"], getsaved(checkpointer, "ψ_∂u∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂u∂z"], getsaved(checkpointer, "ψ_∂u∂z", checkpointer.curr_checkpoint))
            recover!(
                checkpointer,
                recit -> begin
                    backend.forward_onestep_CPML!(
                        model,
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
                        recit,
                        Mxx_bk, Mzz_bk, Mxz_bk;
                        save_trace=false
                    )
                    return ["ucur" => grid.fields["ucur"]]
                end
            )
        end
        # Get velocity fields from saved buffer
        uold_corr = getsaved(checkpointer, "ucur", it - 2).value
        ucur_corr = getsaved(checkpointer, "ucur", it - 1).value
        unew_corr = getsaved(checkpointer, "ucur", it).value

        # Correlate for gradient computation
        backend.correlate_gradients!(grid, uold_corr, ucur_corr, unew_corr, model.dt, model.cpmlparams.freeboundtop)
    end
    # Allocate gradients
    gradient_ρ = zeros(T, grid.size...)
    gradient_λ = zeros(T, grid.size...)
    gradient_μ = zeros(T, grid.size...)
    # Get gradients
    copyto!(gradient_λ, grid.fields["grad_λ"].value)
    copyto!(gradient_μ, grid.fields["grad_μ"].value)
    # Compute chain rules for back interpolations
    gradient_ρ .+= back_interp(model.matprop.interp_method_ρ, model.matprop.ρ, Array(grid.fields["grad_ρ_ihalf"].value), 1) .+
                   back_interp(model.matprop.interp_method_ρ, model.matprop.ρ, Array(grid.fields["grad_ρ_jhalf"].value), 2)
    gradient_μ .+= back_interp(model.matprop.interp_method_μ, model.matprop.μ, Array(grid.fields["grad_μ_ihalf_jhalf"].value), [1, 2])
    # smooth gradient
    @warn "Gradient smoothing not yet implemented"
    # # Compute regularization if needed
    # ∂χ_∂ρ, ∂χ_∂λ, ∂χ_∂μ = (misfit.regularization !== nothing) ? dχ_dm(misfit.regularization, model.matprop) : (0, 0, 0)
    # Return gradients
    return Dict(
        "rho" => gradient_ρ, #.+ ∂χ_∂ρ,
        "lambda" => gradient_λ, #.+ ∂χ_∂λ,
        "mu" => gradient_μ #.+ ∂χ_∂μ
    )
end

# function swgradient_1shot!(
#     ::CPMLBoundaryCondition,
#     model::ElasticIsoCPMLWaveSimulation{T, 2},
#     shot::ExternalForceShot{T, 2},
#     misfit::AbstractMisfit
# )::Dict{String, Array{T, 2}} where {T}
#     # scale source time function
#     srccoeij_ux, srccoeval_ux,
#     srccoeij_uz, srccoeval_uz,
#     reccoeij_ux, reccoeval_ux,
#     reccoeij_uz, reccoeval_uz,
#     scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

#     # Get computational grid, checkpointer and backend
#     grid = model.grid
#     checkpointer = model.checkpointer
#     backend = select_backend(typeof(model), model.runparams.parall)

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

#     # Forward time loop
#     for it in 1:nt
#         # Compute one forward step
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
#             save_trace=true
#         )

#         # Print timestep info
#         printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjforw)

#         # Save checkpoint
#         savecheckpoint!(checkpointer, "σ" => grid.fields["σ"], it)
#         savecheckpoint!(checkpointer, "v" => grid.fields["v"], it)
#         savecheckpoint!(checkpointer, "ψ_∂σ∂x" => grid.fields["ψ_∂σ∂x"], it)
#         savecheckpoint!(checkpointer, "ψ_∂σ∂z" => grid.fields["ψ_∂σ∂z"], it)
#         savecheckpoint!(checkpointer, "ψ_∂v∂x" => grid.fields["ψ_∂v∂x"], it)
#         savecheckpoint!(checkpointer, "ψ_∂v∂z" => grid.fields["ψ_∂v∂z"], it)
#     end

#     @info "Saving seismograms"
#     copyto!(shot.recs.seismograms, traces_bk)

#     @debug "Computing residuals"
#     adjoint_src = backend.Data.Array(dχ_du(misfit, shot.recs))

#     @debug "Computing gradients"
#     # Adjoint time loop (backward in time)
#     for it in nt:-1:1
#         # Compute one adjoint step
#         backend.adjoint_onestep_CPML!(
#             model,
#             reccoeij_ux,
#             reccoeval_ux,
#             reccoeij_uz,
#             reccoeval_uz,
#             adjoint_src,
#             it
#         )
#         # Print timestep info
#         printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjback)

#         # Check if out of save buffer
#         if !issaved(checkpointer, "v", it - 1)
#             @debug @sprintf("Out of save buffer at iteration: %d", it)
#             initrecover!(checkpointer)
#             copyto!(grid.fields["σ"], getsaved(checkpointer, "σ", checkpointer.curr_checkpoint))
#             copyto!(grid.fields["v"], getsaved(checkpointer, "v", checkpointer.curr_checkpoint))
#             copyto!(grid.fields["ψ_∂σ∂x"], getsaved(checkpointer, "ψ_∂σ∂x", checkpointer.curr_checkpoint))
#             copyto!(grid.fields["ψ_∂σ∂z"], getsaved(checkpointer, "ψ_∂σ∂z", checkpointer.curr_checkpoint))
#             copyto!(grid.fields["ψ_∂v∂x"], getsaved(checkpointer, "ψ_∂v∂x", checkpointer.curr_checkpoint))
#             copyto!(grid.fields["ψ_∂v∂z"], getsaved(checkpointer, "ψ_∂v∂z", checkpointer.curr_checkpoint))
#             recover!(
#                 checkpointer,
#                 recit -> begin
#                     backend.forward_onestep_CPML!(
#                         model,
#                         srccoeij_ux,
#                         srccoeval_ux,
#                         srccoeij_uz,
#                         srccoeval_uz,
#                         reccoeij_ux,
#                         reccoeval_ux,
#                         reccoeij_uz,
#                         reccoeval_uz,
#                         srctf_bk,
#                         reduced_buf,
#                         traces_ux_bk_buf,
#                         traces_uz_bk_buf,
#                         traces_bk,
#                         recit;
#                         save_trace=false
#                     )
#                     return ["v" => grid.fields["v"]]
#                 end
#             )
#         end
#         # Get velocity fields from saved buffer
#         v_curr = getsaved(checkpointer, "v", it).value
#         v_old = getsaved(checkpointer, "v", it - 1).value
#         # Correlate for gradient computation
#         backend.correlate_gradients!(grid, v_curr, v_old, model.dt, model.cpmlparams.freeboundtop)
#     end
#     # Allocate gradients
#     gradient_ρ = zeros(T, grid.size...)
#     gradient_λ = zeros(T, grid.size...)
#     gradient_μ = zeros(T, grid.size...)
#     # Get gradients
#     copyto!(gradient_λ, grid.fields["grad_λ"].value)
#     copyto!(gradient_μ, grid.fields["grad_μ"].value)
#     # Compute chain rules for back interpolations
#     gradient_ρ .+= back_interp(model.matprop.interp_method_ρ, model.matprop.ρ, Array(grid.fields["grad_ρ_ihalf"].value), 1) .+
#                    back_interp(model.matprop.interp_method_ρ, model.matprop.ρ, Array(grid.fields["grad_ρ_jhalf"].value), 2)
#     gradient_μ .+= back_interp(model.matprop.interp_method_μ, model.matprop.μ, Array(grid.fields["grad_μ_ihalf_jhalf"].value), [1, 2])
#    # smooth gradient
#    @warn "Gradient smoothing not yet implemented"
#    backend.smooth_gradient!(gradient_ρ, possrcs, model.smooth_radius)
#    backend.smooth_gradient!(gradient_λ, possrcs, model.smooth_radius)
#    backend.smooth_gradient!(gradient_μ, possrcs, model.smooth_radius)
# Compute regularization if needed
#     ∂χ_∂ρ, ∂χ_∂λ, ∂χ_∂μ = (misfit.regularization !== nothing) ? dχ_dm(misfit.regularization, model.matprop) : (0, 0, 0)
#     # Return gradients
#     return Dict(
#         "rho" => gradient_ρ .+ ∂χ_∂ρ,
#         "lambda" => gradient_λ .+ ∂χ_∂λ,
#         "mu" => gradient_μ .+ ∂χ_∂μ
#     )
# end
