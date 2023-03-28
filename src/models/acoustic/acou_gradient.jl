

@views function swgradient_1shot!(
    model::Acoustic_CD_CPML_WaveSimul, # includes types Acoustic_CD_CPML_WaveSimul{1}, ...{2} and ...{3}
    backend::Module,
    possrcs, posrecs, srctf, traces, observed, invcov ;
    check_freq=nothing
)
    # Numerics
    N = length(model.ns)
    nt = model.nt
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(model.ns...)
    pcur = backend.zeros(model.ns...)
    pnew = backend.zeros(model.ns...)
    fact_a = backend.Data.Array( model.fact )
    # Initialize adjoint arrays
    adjold = backend.zeros(model.ns...)
    adjcur = backend.zeros(model.ns...)
    adjnew = backend.zeros(model.ns...)
    # Initialize CPML arrays (forward and adjoint)
    ψ = []
    ξ = []
    ψ_adj = []
    ξ_adj = []
    for i = 1:N
        ψ_ns = [model.ns...]
        ξ_ns = [model.ns...]
        ψ_ns[i] = halo+1
        ξ_ns[i] = halo
        append!(ψ, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
        append!(ξ, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
        append!(ψ_adj, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
        append!(ξ_adj, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
    end
    # Wrap CPML coefficient arrays
    a_coeffs = []
    b_K_coeffs = []
    for i = 1:N
        append!(a_coeffs, backend.Data.Array.([
            model.cpmlcoeffs[i].a_l,
            model.cpmlcoeffs[i].a_r,
            model.cpmlcoeffs[i].a_hl,
            model.cpmlcoeffs[i].a_hr
        ]))
        append!(b_K_coeffs, backend.Data.Array.([
            model.cpmlcoeffs[i].b_K_l,
            model.cpmlcoeffs[i].b_K_r,
            model.cpmlcoeffs[i].b_K_hl,
            model.cpmlcoeffs[i].b_K_hr
        ]))
    end
    # Add empty row to source time functions (fix for last timestep)
    srctf = [srctf; zeros(1, size(srctf,2))]
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )
    observed_a = backend.Data.Array( observed )
    invcov_a = backend.Data.Array( invcov )

    # Checkpointing setup
    if check_freq !== nothing   # checkpointing enabled
        # Time step of last checkpoint
        last_checkpoint = floor(Int, nt / check_freq) * check_freq
        # Checkpointing arrays
        save_buffer = backend.zeros(model.ns..., check_freq + 2)     # pressure window buffer
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints
        checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints
        checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints
        # Save initial conditions as first checkpoint
        checkpoints[-1] = copy(pold)
        checkpoints[0] = copy(pcur)
        checkpoints_ψ[0] = copy.(ψ)
        checkpoints_ξ[0] = copy.(ξ)
    else    # no checkpointing
        last_checkpoint = 1                                 # simulate a checkpoint at time step 1 (so buffer will start from 0)
        save_buffer = backend.zeros(model.ns..., nt+2)      # save all timesteps (from 0 to n+1 so n+2)
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
        checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
        checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
        # Save time step 0 into buffer
        save_buffer[fill(Colon(), N)..., 1] .= pcur
    end
    # Residuals arrays
    residuals_a = similar(traces_a)
    # Gradient arrays
    curgrad = backend.zeros(model.ns...)

    # Forward time loop (nt+1 timesteps)
    for it = 1:nt+1
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, model.gridspacing..., halo,
            ψ..., ξ..., a_coeffs..., b_K_coeffs...,
            possrcs_a, srctf_a, posrecs_a, traces_a, it;
            save_trace=(it <= nt)   # do not save trace for last timestep
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s]", it, model.dt*it)
        end
        # Save checkpoint
        if check_freq !== nothing && it % check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current and last timestep pressure
            checkpoints[it] = copy(pcur)
            checkpoints[it-1] = copy(pold)
            # Also save CPML arrays
            checkpoints_ψ[it] = copy.(ψ)
            checkpoints_ξ[it] = copy.(ξ)
        end
        # Start populating save buffer just before last checkpoint
        if it >= last_checkpoint-1
            save_buffer[fill(Colon(), N)..., it - (last_checkpoint-1) + 1] .= pcur
        end
    end

    @debug "Computing residuals"
    # Compute residuals
    mul!(residuals_a, invcov_a, traces_a - observed_a)
    # Prescale residuals (fact = vel^2 * dt^2)
    backend.prescale_residuals!(residuals_a, possrcs_a, fact_a)

    @debug "Computing gradients"
    # Current checkpoint
    curr_checkpoint = last_checkpoint
    # Adjoint time loop (backward in time)
    for it = nt:-1:1
        # Compute one adjoint step
        adjold, adjcur, adjnew = backend.forward_onestep_CPML!(
            adjold, adjcur, adjnew, fact_a, model.gridspacing..., halo,
            ψ_adj..., ξ_adj..., a_coeffs..., b_K_coeffs...,
            posrecs_a, residuals_a, nothing, nothing, it;   # adjoint sources positions are receivers
            save_trace=false   # do not save traces in adjoint time loop
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
            curr_checkpoint -= check_freq
            # Shift start of save buffer
            save_buffer[fill(Colon(), N)..., 1] .= checkpoints[curr_checkpoint-1]
            save_buffer[fill(Colon(), N)..., 2] .= checkpoints[curr_checkpoint]
            # Shift end of save buffer
            save_buffer[fill(Colon(), N)..., end-1] .= checkpoints[old_checkpoint-1]
            save_buffer[fill(Colon(), N)..., end] .= checkpoints[old_checkpoint]
            # Recover pressure and CPML arrays from current checkpoint
            pold .= checkpoints[curr_checkpoint-1]
            pcur .= checkpoints[curr_checkpoint]
            pnew .= 0.0
            for i in eachindex(checkpoints_ψ[curr_checkpoint])
                ψ[i] .= checkpoints_ψ[curr_checkpoint][i]
            end
            for i in eachindex(checkpoints_ξ[curr_checkpoint])
                ξ[i] .= checkpoints_ξ[curr_checkpoint][i]
            end
            # Forward recovery time loop
            for recit = curr_checkpoint+1:(old_checkpoint-1)-1
                pold, pcur, pnew = backend.forward_onestep_CPML!(
                    pold, pcur, pnew, fact_a, model.gridspacing..., halo,
                    ψ..., ξ..., a_coeffs..., b_K_coeffs...,
                    possrcs_a, srctf_a, nothing, nothing, recit;
                    save_trace=false   # do not save trace in recovery time loop
                )
                if recit % model.infoevery == 0
                    @debug @sprintf("Recovering iteration: %d", recit)
                end
                # Save recovered pressure in save buffer
                save_buffer[fill(Colon(), N)..., recit - (curr_checkpoint+1) + 3] .= pcur
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = save_buffer[fill(Colon(), N)..., it - curr_checkpoint + 1]
        pold_corr = save_buffer[fill(Colon(), N)..., it - curr_checkpoint + 2]
        pveryold_corr = save_buffer[fill(Colon(), N)..., it - curr_checkpoint + 3]
        # Correlate for gradient computation
        backend.correlate_gradient!(curgrad, adjcur, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end

    return Array( curgrad )
end
