swgradient_1shot!(model::AcousticWaveSimul, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

@views function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimul{N},
    possrcs,
    posrecs,
    srctf,
    traces,
    observed,
    invcov
)::Array{<:Real} where {N}
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
    # Add empty row to source time functions (fix for last timestep)
    srctf = [srctf; zeros(1, size(srctf, 2))]
    # Wrap sources and receivers arrays
    possrcs_a = model.backend.Data.Array(possrcs)
    posrecs_a = model.backend.Data.Array(posrecs)
    srctf_a = model.backend.Data.Array(srctf)
    traces_a = model.backend.Data.Array(traces)
    observed_a = model.backend.Data.Array(observed)
    invcov_a = model.backend.Data.Array(invcov)
    # Residuals arrays
    residuals_a = similar(traces_a)
    # Reset wavesim
    reset!(model)

    # Forward time loop (nt+1 timesteps)
    for it in 1:(nt+1)
        # Compute one forward step
        pold, pcur, pnew = model.backend.forward_onestep_CPML!(
            pold, pcur, pnew, model.fact,
            model.gridspacing..., model.halo,
            model.ψ..., model.ξ..., model.a_coeffs..., model.b_K_coeffs...,
            possrcs_a, srctf_a, posrecs_a, traces_a, it;
            save_trace=(it <= nt)
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s]", it, model.dt * it)
        end
        # Save checkpoint
        if model.check_freq !== nothing && it % model.check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current and last timestep pressure
            model.checkpoints[it] .= pcur
            model.checkpoints[it-1] .= pold
            # Also save CPML arrays
            for i in eachindex(model.ψ)
                model.checkpoints_ψ[it][i] .= model.ψ[i]
            end
            for i in eachindex(model.ξ)
                model.checkpoints_ξ[it][i] .= model.ξ[i]
            end
        end
        # Start populating save buffer just before last checkpoint
        if it >= model.last_checkpoint - 1
            model.save_buffer[fill(Colon(), N)..., it-(model.last_checkpoint-1)+1] .= pcur
        end
    end

    # Save traces
    traces .= Array(traces_a)

    @debug "Computing residuals"
    # Compute residuals
    mul!(residuals_a, invcov_a, traces_a - observed_a)
    # Prescale residuals (fact = vel^2 * dt^2)
    model.backend.prescale_residuals!(residuals_a, possrcs_a, model.fact)

    @debug "Computing gradients"
    # Current checkpoint
    curr_checkpoint = model.last_checkpoint
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        adjold, adjcur, adjnew = model.backend.forward_onestep_CPML!(
            adjold, adjcur, adjnew,
            model.fact, model.gridspacing..., model.halo,
            model.ψ_adj..., model.ξ_adj..., model.a_coeffs..., model.b_K_coeffs...,
            posrecs_a, residuals_a, nothing, nothing, it;   # adjoint sources positions are receivers
            save_trace=false
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Backward iteration: %d", it)
        end
        # Check if out of save buffer
        if it < curr_checkpoint
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            # Shift last checkpoint
            old_checkpoint = curr_checkpoint
            curr_checkpoint -= model.check_freq
            # Shift start of save buffer
            model.save_buffer[fill(Colon(), N)..., 1] .= model.checkpoints[curr_checkpoint-1]
            model.save_buffer[fill(Colon(), N)..., 2] .= model.checkpoints[curr_checkpoint]
            # Shift end of save buffer
            model.save_buffer[fill(Colon(), N)..., end-1] .= model.checkpoints[old_checkpoint-1]
            model.save_buffer[fill(Colon(), N)..., end] .= model.checkpoints[old_checkpoint]
            # Recover pressure and CPML arrays from current checkpoint
            pold .= model.checkpoints[curr_checkpoint-1]
            pcur .= model.checkpoints[curr_checkpoint]
            pnew .= 0.0
            for i in eachindex(model.checkpoints_ψ[curr_checkpoint])
                model.ψ[i] .= model.checkpoints_ψ[curr_checkpoint][i]
            end
            for i in eachindex(model.checkpoints_ξ[curr_checkpoint])
                model.ξ[i] .= model.checkpoints_ξ[curr_checkpoint][i]
            end
            # Forward recovery time loop
            for recit in (curr_checkpoint+1):((old_checkpoint-1)-1)
                pold, pcur, pnew = model.backend.forward_onestep_CPML!(
                    pold, pcur, pnew, model.fact,
                    model.gridspacing..., model.halo,
                    model.ψ..., model.ξ..., model.a_coeffs..., model.b_K_coeffs...,
                    possrcs_a, srctf_a, nothing, nothing, recit;
                    save_trace=false
                )
                if recit % model.infoevery == 0
                    @debug @sprintf("Recovering iteration: %d", recit)
                end
                # Save recovered pressure in save buffer
                model.save_buffer[fill(Colon(), N)..., recit-(curr_checkpoint+1)+3] .= pcur
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = model.save_buffer[fill(Colon(), N)..., it-curr_checkpoint+1]
        pold_corr = model.save_buffer[fill(Colon(), N)..., it-curr_checkpoint+2]
        pveryold_corr = model.save_buffer[fill(Colon(), N)..., it-curr_checkpoint+3]
        # Correlate for gradient computation
        model.backend.correlate_gradient!(model.curgrad, adjcur, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end

    # rescale gradient
    gradient = Array(model.curgrad)
    gradient = (2.0 ./ (model.matprop.vp .^ 3)) .* gradient
    return gradient
end
