swgradient_1shot!(model::ElasticWaveSimulation, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::MomentTensorShot{T, 2, MomentTensor2D{T}},
    misfit::AbstractMisfit
)::Dict{String, Array{T, 2}} where {T}
    # scale source time function
    srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz,
    reccoeij_vx, reccoeval_vx, reccoeij_vz, reccoeval_vz,
    scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)
    # moment tensors
    momtens = shot.srcs.momtens

    # Get computational grid, checkpointer and backend
    grid = model.grid
    checkpointer = model.checkpointer
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    srccoeij_xx = [backend.Data.Array(srccoeij_xx[i]) for i in eachindex(srccoeij_xx)]
    srccoeval_xx = [backend.Data.Array(srccoeval_xx[i]) for i in eachindex(srccoeval_xx)]
    srccoeij_xz = [backend.Data.Array(srccoeij_xz[i]) for i in eachindex(srccoeij_xz)]
    srccoeval_xz = [backend.Data.Array(srccoeval_xz[i]) for i in eachindex(srccoeval_xz)]
    reccoeij_vx = [backend.Data.Array(reccoeij_vx[i]) for i in eachindex(reccoeij_vx)]
    reccoeval_vx = [backend.Data.Array(reccoeval_vx[i]) for i in eachindex(reccoeval_vx)]
    reccoeij_vz = [backend.Data.Array(reccoeij_vz[i]) for i in eachindex(reccoeij_vz)]
    reccoeval_vz = [backend.Data.Array(reccoeval_vz[i]) for i in eachindex(reccoeval_vz)]

    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

    ## ONLY 2D for now!!!
    nsrcs = length(momtens)
    Mxx_bk = backend.Data.Array([momtens[i].Mxx for i in 1:nsrcs])
    Mzz_bk = backend.Data.Array([momtens[i].Mzz for i in 1:nsrcs])
    Mxz_bk = backend.Data.Array([momtens[i].Mxz for i in 1:nsrcs])

    # Reset wavesim
    reset!(model)

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(
            model,
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
            save_trace=true
        )

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
        # Save checkpoint
        savecheckpoint!(checkpointer, "σ" => grid.fields["σ"], it)
        savecheckpoint!(checkpointer, "v" => grid.fields["v"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂x" => grid.fields["ψ_∂σ∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂z" => grid.fields["ψ_∂σ∂z"], it)
        savecheckpoint!(checkpointer, "ψ_∂v∂x" => grid.fields["ψ_∂v∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂v∂z" => grid.fields["ψ_∂v∂z"], it)
    end

    @info "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    residuals_bk = backend.Data.Array(dχ_du(misfit, shot.recs))

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(
            model,
            reccoeij_vx,
            reccoeval_vx,
            reccoeij_vz,
            reccoeval_vz,
            residuals_bk,
            it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if !issaved(checkpointer, "v", it - 1)
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            initrecover!(checkpointer)
            copyto!(grid.fields["σ"], getsaved(checkpointer, "σ", checkpointer.curr_checkpoint))
            copyto!(grid.fields["v"], getsaved(checkpointer, "v", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂x"], getsaved(checkpointer, "ψ_∂σ∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂z"], getsaved(checkpointer, "ψ_∂σ∂z", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂v∂x"], getsaved(checkpointer, "ψ_∂v∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂v∂z"], getsaved(checkpointer, "ψ_∂v∂z", checkpointer.curr_checkpoint))
            recover!(
                checkpointer,
                recit -> begin
                    backend.forward_onestep_CPML!(
                        model,
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
                        recit,
                        Mxx_bk, Mzz_bk, Mxz_bk;
                        save_trace=false
                    )
                    return ["v" => grid.fields["v"]]
                end
            )
        end
        # Get velocity fields from saved buffer
        v_curr = getsaved(checkpointer, "v", it).value
        v_old = getsaved(checkpointer, "v", it - 1).value
        # Correlate for gradient computation
        backend.correlate_gradients!(grid, v_curr, v_old, model.dt, model.cpmlparams.freeboundtop)
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
    # TODO smooth gradients
    # Compute regularization if needed
    ∂χ_∂ρ, ∂χ_∂λ, ∂χ_∂μ = (misfit.regularization !== nothing) ? dχ_dm(misfit.regularization, model.matprop) : (0, 0, 0)
    # Return gradients
    return Dict(
        "rho" => gradient_ρ .+ ∂χ_∂ρ,
        "lambda" => gradient_λ .+ ∂χ_∂λ,
        "mu" => gradient_μ .+ ∂χ_∂μ
    )
end

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::ElasticIsoCPMLWaveSimulation{T, 2},
    shot::ExternalForceShot{T, 2},
    misfit::AbstractMisfit
)::Dict{String, Array{T, 2}} where {T}
    # scale source time function
    srccoeij_vx, srccoeval_vx,
    srccoeij_vz, srccoeval_vz,
    reccoeij_vx, reccoeval_vx,
    reccoeij_vz, reccoeval_vz,
    scal_srctf = possrcrec_scaletf(model, shot; sincinterp=model.sincinterp)

    # Get computational grid, checkpointer and backend
    grid = model.grid
    checkpointer = model.checkpointer
    backend = select_backend(typeof(model), model.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    srccoeij_vx = [backend.Data.Array(srccoeij_vx[i]) for i in eachindex(srccoeij_vx)]
    srccoeval_vx = [backend.Data.Array(srccoeval_vx[i]) for i in eachindex(srccoeval_vx)]
    srccoeij_vz = [backend.Data.Array(srccoeij_vz[i]) for i in eachindex(srccoeij_vz)]
    srccoeval_vz = [backend.Data.Array(srccoeval_vz[i]) for i in eachindex(srccoeval_vz)]
    reccoeij_vx = [backend.Data.Array(reccoeij_vx[i]) for i in eachindex(reccoeij_vx)]
    reccoeval_vx = [backend.Data.Array(reccoeval_vx[i]) for i in eachindex(reccoeval_vx)]
    reccoeij_vz = [backend.Data.Array(reccoeij_vz[i]) for i in eachindex(reccoeij_vz)]
    reccoeval_vz = [backend.Data.Array(reccoeval_vz[i]) for i in eachindex(reccoeval_vz)]

    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.zeros(T, size(shot.recs.seismograms))

    # Reset wavesim
    reset!(model)

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
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
            save_trace=true
        )

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
        # Save checkpoint
        savecheckpoint!(checkpointer, "σ" => grid.fields["σ"], it)
        savecheckpoint!(checkpointer, "v" => grid.fields["v"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂x" => grid.fields["ψ_∂σ∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂σ∂z" => grid.fields["ψ_∂σ∂z"], it)
        savecheckpoint!(checkpointer, "ψ_∂v∂x" => grid.fields["ψ_∂v∂x"], it)
        savecheckpoint!(checkpointer, "ψ_∂v∂z" => grid.fields["ψ_∂v∂z"], it)
    end

    @info "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    residuals_bk = backend.Data.Array(dχ_du(misfit, shot.recs))

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(
            model,
            reccoeij_vx,
            reccoeval_vx,
            reccoeij_vz,
            reccoeval_vz,
            residuals_bk,
            it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if !issaved(checkpointer, "v", it - 1)
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            initrecover!(checkpointer)
            copyto!(grid.fields["σ"], getsaved(checkpointer, "σ", checkpointer.curr_checkpoint))
            copyto!(grid.fields["v"], getsaved(checkpointer, "v", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂x"], getsaved(checkpointer, "ψ_∂σ∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂σ∂z"], getsaved(checkpointer, "ψ_∂σ∂z", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂v∂x"], getsaved(checkpointer, "ψ_∂v∂x", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ_∂v∂z"], getsaved(checkpointer, "ψ_∂v∂z", checkpointer.curr_checkpoint))
            recover!(
                checkpointer,
                recit -> begin
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
                        recit;
                        save_trace=false
                    )
                    return ["v" => grid.fields["v"]]
                end
            )
        end
        # Get velocity fields from saved buffer
        v_curr = getsaved(checkpointer, "v", it).value
        v_old = getsaved(checkpointer, "v", it - 1).value
        # Correlate for gradient computation
        backend.correlate_gradients!(grid, v_curr, v_old, model.dt, model.cpmlparams.freeboundtop)
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
    # TODO smooth gradients
    # Compute regularization if needed
    ∂χ_∂ρ, ∂χ_∂λ, ∂χ_∂μ = (misfit.regularization !== nothing) ? dχ_dm(misfit.regularization, model.matprop) : (0, 0, 0)
    # Return gradients
    return Dict(
        "rho" => gradient_ρ .+ ∂χ_∂ρ,
        "lambda" => gradient_λ .+ ∂χ_∂λ,
        "mu" => gradient_μ .+ ∂χ_∂μ
    )
end