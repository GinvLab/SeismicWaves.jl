swgradient_1shot!(model::AcousticWaveSimul, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimul{T,N},
    shot::Shot,
    misfit
)::Dict{String, Array{T,N}} where {T,N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    grid = model.grid
    checkpointer = model.checkpointer
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

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Forward iteration: %d, simulation time: %g [s]", it, model.dt * it)
        end
        # Save checkpoint
        savecheckpoint!(checkpointer, "pcur" => grid.fields["pcur"], it)
        savecheckpoint!(checkpointer, "ψ" => grid.fields["ψ"], it)
        savecheckpoint!(checkpointer, "ξ" => grid.fields["ξ"], it)
    end

    @info "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    residuals_bk = backend.Data.Array(dχ_du(misfit, shot.recs))

    # Prescale residuals (fact = vel^2 * dt^2)
    backend.prescale_residuals!(residuals_bk, posrecs_bk, grid.fields["fact"].value)

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(grid, posrecs_bk, residuals_bk, it)
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if !issaved(checkpointer, "pcur", it - 2)
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            initrecover!(checkpointer)
            copyto!(grid.fields["pold"], getsaved(checkpointer, "pcur", checkpointer.curr_checkpoint - 1))
            copyto!(grid.fields["pcur"], getsaved(checkpointer, "pcur", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ"], getsaved(checkpointer, "ψ", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ξ"], getsaved(checkpointer, "ξ", checkpointer.curr_checkpoint))
            recover!(
                checkpointer,
                recit -> begin
                    backend.forward_onestep_CPML!(grid, possrcs_bk, srctf_bk, nothing, nothing, recit; save_trace=false)
                    return ["pcur" => grid.fields["pcur"]]
                end
            )
        end
        # Get pressure fields from saved buffer
        pcur_corr = getsaved(checkpointer, "pcur", it - 2).value
        pold_corr = getsaved(checkpointer, "pcur", it - 1).value
        pveryold_corr = getsaved(checkpointer, "pcur", it).value
        # Correlate for gradient computation
        backend.correlate_gradient!(grid.fields["grad_vp"].value, grid.fields["adjcur"].value, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end
    # get gradient
    gradient = Array(grid.fields["grad_vp"].value)
    # smooth gradient
    backend.smooth_gradient!(gradient, possrcs, model.smooth_radius)
    # rescale gradient
    gradient .= (2.0 ./ (model.matprop.vp .^ 3)) .* gradient
    # add regularization if needed
    if misfit.regularization !== nothing
        gradient .+= dχ_dm(misfit.regularization, model.matprop)
    end
    return Dict("vp" => gradient)
end

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimul{T,N},
    shot::Shot,
    misfit
)::Dict{String, Array{T,N}} where {T,N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    # Numerics
    nt = model.nt
    # Initialize pressure and velocities arrays
    pcur = model.pcur
    vcur = model.vcur
    # Initialize adjoint arrays
    adjpcur = model.adjpcur
    adjvcur = model.adjvcur
    # Wrap sources and receivers arrays
    possrcs_bk = model.backend.Data.Array(possrcs)
    posrecs_bk = model.backend.Data.Array(posrecs)
    srctf_bk = model.backend.Data.Array(scal_srctf)
    traces_bk = model.backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        model.backend.forward_onestep_CPML!(
            pcur, vcur..., model.fact_m0, model.fact_m1_stag...,
            model.gridspacing..., model.halo,
            model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
            possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Forward iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt * it, maximum(abs.(Array(pcur))))
        end
        # Save checkpoint
        if model.check_freq !== nothing && it % model.check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current pressure and velocities
            copyto!(model.checkpoints[it], pcur)
            for i in eachindex(vcur)
                copyto!(model.checkpoints_v[it][i], vcur[i])
            end
            # Also save CPML arrays
            for i in eachindex(model.ψ)
                copyto!(model.checkpoints_ψ[it][i], model.ψ[i])
            end
            for i in eachindex(model.ξ)
                copyto!(model.checkpoints_ξ[it][i], model.ξ[i])
            end
        end
        # Start populating save buffer at last checkpoint
        if it >= model.last_checkpoint
            copyto!(model.save_buffer[fill(Colon(), N)..., it-model.last_checkpoint+1], pcur)
        end
    end

    @info "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    residuals_bk = model.backend.Data.Array(dχ_du(misfit, shot.recs))

    # Prescale residuals (fact = vel^2 * rho * dt)
    model.backend.prescale_residuals!(residuals_bk, posrecs_bk, model.fact_m0)

    @debug "Computing gradients"
    # Current checkpoint
    curr_checkpoint = model.last_checkpoint
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        model.backend.backward_onestep_CPML!(
            adjpcur, adjvcur..., model.fact_m0, model.fact_m1_stag...,
            model.gridspacing..., model.halo,
            model.ψ_adj..., model.ξ_adj..., model.a_coeffs..., model.b_coeffs...,
            posrecs_bk, residuals_bk, it; # adjoint sources positions are receivers
        )
        # Print timestep info
        if it % model.infoevery == 0
            @info @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if (it - 1) < curr_checkpoint
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            # Shift last checkpoint
            old_checkpoint = curr_checkpoint
            curr_checkpoint -= model.check_freq
            # Shift start of save buffer
            copyto!(model.save_buffer[fill(Colon(), N)..., 1], model.checkpoints[curr_checkpoint])
            # Shift end of save save_buffer
            copyto!(model.save_buffer[fill(Colon(), N)..., end], model.checkpoints[old_checkpoint])
            # Recover pressure, velocities and CPML arrays from current checkpoint
            copyto!(pcur, model.checkpoints[curr_checkpoint])
            for i in eachindex(model.checkpoints_v[curr_checkpoint])
                copyto!(vcur[i], model.checkpoints_v[curr_checkpoint][i])
            end
            for i in eachindex(model.checkpoints_ψ[curr_checkpoint])
                copyto!(model.ψ[i], model.checkpoints_ψ[curr_checkpoint][i])
            end
            for i in eachindex(model.checkpoints_ξ[curr_checkpoint])
                copyto!(model.ξ[i], model.checkpoints_ξ[curr_checkpoint][i])
            end
            # Forward recovery time loop
            for recit in (curr_checkpoint+1):(old_checkpoint-1)
                model.backend.forward_onestep_CPML!(
                    pcur, vcur..., model.fact_m0, model.fact_m1_stag...,
                    model.gridspacing..., model.halo,
                    model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
                    possrcs_bk, srctf_bk, nothing, nothing, recit;
                    save_trace=false
                )
                # Save recovered pressure in save buffer
                copyto!(model.save_buffer[fill(Colon(), N)..., recit-(curr_checkpoint+1)+2], pcur)
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = model.save_buffer[fill(Colon(), N)..., (it-curr_checkpoint)+1]
        pcur_old = model.save_buffer[fill(Colon(), N)..., (it-curr_checkpoint)]
        # Correlate for gradient computation
        model.backend.correlate_gradient_m0!(model.curgrad_m0, adjpcur, pcur_corr, pcur_old, model.dt)
        model.backend.correlate_gradient_m1!(model.curgrad_m1_stag, adjvcur, pcur_corr, model.gridspacing)
    end
    # Allocate gradients
    gradient_m0 = zeros(T, model.ns...)
    gradient_m1 = zeros(T, model.ns...)
    # Get gradients
    copyto!(gradient_m0, model.curgrad_m0)
    for i in eachindex(model.curgrad_m1_stag)
        # Accumulate and interpolate staggered gradients
        gradient_m1[CartesianIndices(Tuple(j == i ? (2:model.ns[j]-1) : (1:model.ns[j]) for j in 1:N))] .+= interp(model.matprop.interp_method, Array(model.curgrad_m1_stag[i]), i)
    end
    # Smooth gradients
    model.backend.smooth_gradient!(gradient_m0, possrcs, model.smooth_radius)
    model.backend.smooth_gradient!(gradient_m1, possrcs, model.smooth_radius)
    # compute regularization if needed
    dχ_dvp, dχ_drho = (misfit.regularization !== nothing) ? dχ_dm(misfit.regularization, model.matprop) : (0, 0)
    # Rescale gradients with respect to material properties (chain rule)
    return Dict(
        "vp"  => .-2.0 .* gradient_m0 ./ (model.matprop.vp .^ 3 .* model.matprop.rho) .+ dχ_dvp,                                     # grad wrt vp
        "rho" => .-gradient_m0 ./ (model.matprop.vp .^ 2 .* model.matprop.rho .^ 2) .- gradient_m1 ./ model.matprop.rho .+ dχ_drho   # grad wrt rho
    )
end
