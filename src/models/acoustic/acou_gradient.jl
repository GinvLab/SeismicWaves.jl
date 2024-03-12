swgradient_1shot!(model::AcousticWaveSimul, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimul{N},
    shot::Shot,
    # possrcs,
    # posrecs,
    # srctf,
    # recs,
    misfit
    )::Array{<:Real} where {N}

    # scale source time function, etc.
    possrcs,posrecs,scal_srctf = possrcrec_scaletf(model,shot)

    # Numerics
    nt = model.nt
    # Initialize pressure and factors arrays
    pold = model.pold
    pcur = model.pcur
    pnew = model.pnew
    # Initialize adjoint arrays
    adjold = model.adjold
    adjcur = model.adjcur
    adjnew = model.adjnew
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
        pold, pcur, pnew = model.backend.forward_onestep_CPML!(
            pold, pcur, pnew, model.fact,
            model.gridspacing..., model.halo,
            model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
            possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s]", it, model.dt * it)
        end
        # Save checkpoint
        if model.check_freq !== nothing && it % model.check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current and last timestep pressure
            copyto!(model.checkpoints[it], pcur)
            copyto!(model.checkpoints[it-1], pold)
            # Also save CPML arrays
            for i in eachindex(model.ψ)
                copyto!(model.checkpoints_ψ[it][i], model.ψ[i])
            end
            for i in eachindex(model.ξ)
                copyto!(model.checkpoints_ξ[it][i], model.ξ[i])
            end
        end
        # Start populating save buffer just before last checkpoint
        if it >= model.last_checkpoint - 1
            copyto!(model.save_buffer[fill(Colon(), N)..., it-(model.last_checkpoint-1)+1], pcur)
        end
    end

    @info "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    residuals_bk = model.backend.Data.Array(dχ_du(misfit, shot.recs))

    # Prescale residuals (fact = vel^2 * dt^2)
    model.backend.prescale_residuals!(residuals_bk, posrecs_bk, model.fact)

    @debug "Computing gradients"
    # Current checkpoint
    curr_checkpoint = model.last_checkpoint
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        adjold, adjcur, adjnew = model.backend.forward_onestep_CPML!(
            adjold, adjcur, adjnew, model.fact,
            model.gridspacing..., model.halo,
            model.ψ_adj..., model.ξ_adj..., model.a_coeffs..., model.b_coeffs...,
            posrecs_bk, residuals_bk, nothing, nothing, it;   # adjoint sources positions are receivers
            save_trace=false
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if (it - 1) < curr_checkpoint
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            # Shift last checkpoint
            old_checkpoint = curr_checkpoint
            curr_checkpoint -= model.check_freq
            # Shift start of save buffer
            copyto!(model.save_buffer[fill(Colon(), N)..., 1], model.checkpoints[curr_checkpoint-1])
            copyto!(model.save_buffer[fill(Colon(), N)..., 2], model.checkpoints[curr_checkpoint])
            # Shift end of save buffer
            copyto!(model.save_buffer[fill(Colon(), N)..., end-1], model.checkpoints[old_checkpoint-1])
            copyto!(model.save_buffer[fill(Colon(), N)..., end], model.checkpoints[old_checkpoint])
            # Recover pressure and CPML arrays from current checkpoint
            copyto!(pold, model.checkpoints[curr_checkpoint-1])
            copyto!(pcur, model.checkpoints[curr_checkpoint])
            pnew .= 0.0
            for i in eachindex(model.checkpoints_ψ[curr_checkpoint])
                copyto!(model.ψ[i], model.checkpoints_ψ[curr_checkpoint][i])
            end
            for i in eachindex(model.checkpoints_ξ[curr_checkpoint])
                copyto!(model.ξ[i], model.checkpoints_ξ[curr_checkpoint][i])
            end
            # Forward recovery time loop
            for recit in (curr_checkpoint+1):((old_checkpoint-1)-1)
                pold, pcur, pnew = model.backend.forward_onestep_CPML!(
                    pold, pcur, pnew, model.fact,
                    model.gridspacing..., model.halo,
                    model.ψ..., model.ξ..., model.a_coeffs..., model.b_coeffs...,
                    possrcs_bk, srctf_bk, nothing, nothing, recit;
                    save_trace=false
                )
                if recit % model.infoevery == 0
                    @debug @sprintf("Recovering iteration: %d", recit)
                end
                # Save recovered pressure in save buffer
                copyto!(model.save_buffer[fill(Colon(), N)..., recit-(curr_checkpoint+1)+3], pcur)
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = model.save_buffer[fill(Colon(), N)..., ((it-1)-curr_checkpoint)+1]
        pold_corr = model.save_buffer[fill(Colon(), N)..., ((it-1)-curr_checkpoint)+2]
        pveryold_corr = model.save_buffer[fill(Colon(), N)..., ((it-1)-curr_checkpoint)+3]
        # Correlate for gradient computation
        model.backend.correlate_gradient!(model.curgrad, adjcur, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end
    # get gradient
    gradient = Array(model.curgrad)
    # smooth gradient
    model.backend.smooth_gradient!(gradient, possrcs, model.smooth_radius)
    # rescale gradient
    gradient .= (2.0 ./ (model.matprop.vp .^ 3)) .* gradient
    # add regularization if needed
    if misfit.regularization !== nothing
        gradient .+= dχ_dm(misfit.regularization, model.matprop)
    end
    return gradient
end

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimul{N},
    shot::Shot,
    # possrcs,
    # posrecs,
    # srctf,
    # recs,
    misfit
    )::Array{<:Real} where {N}

    # scale source time function, etc.
    possrcs,posrecs,scal_srctf = possrcrec_scaletf(model,shot)

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
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt * it, maximum(abs.(Array(pcur))))
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
            copyto!(model.save_buffer[fill(Colon(), N)..., it - model.last_checkpoint + 1], pcur)
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
            @debug @sprintf("Backward iteration: %d", it)
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
                if recit % model.infoevery == 0
                    @debug @sprintf("Recover forward iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", recit, model.dt * recit, maximum(abs.(Array(pcur))))
                end
                # Save recovered pressure in save buffer
                copyto!(model.save_buffer[fill(Colon(), N)..., recit - (curr_checkpoint+1) + 2], pcur)
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = model.save_buffer[fill(Colon(), N)..., (it - curr_checkpoint)+1]
        pcur_old = model.save_buffer[fill(Colon(), N)..., (it - curr_checkpoint)]
        # Correlate for gradient computation
        model.backend.correlate_gradient_m0!(model.curgrad_m0, adjpcur, pcur_corr, pcur_old, model.dt)
        model.backend.correlate_gradient_m1!(model.curgrad_m1_stag, adjvcur, pcur_corr, model.gridspacing)
    end
    # Allocate gradients
    gradient_m0 = zeros(model.ns...)
    gradient_m1 = zeros(model.ns...)
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
    return reshape(hcat(
                .-2.0 .* gradient_m0 ./ (model.matprop.vp .^ 3 .* model.matprop.rho) .+ dχ_dvp,                                     # grad wrt vp
                .-gradient_m0 ./ (model.matprop.vp .^ 2 .* model.matprop.rho .^ 2) .- gradient_m1 ./ model.matprop.rho .+ dχ_drho   # grad wrt rho
            ),
            model.totgrad_size...
        )
end
