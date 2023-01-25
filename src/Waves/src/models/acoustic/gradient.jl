@views function gradient!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, observed, invcov, backend;
    check_freq::Integer = 100
)
    # Numerics
    nt = model.nt
    nx = model.nx
    dx = model.dx
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx)
    pcur = backend.zeros(nx)
    pnew = backend.zeros(nx)
    fact_a = backend.Data.Array( model.fact )
    # Initialize adjoint arrays
    adjold = backend.zeros(nx)
    adjcur = backend.zeros(nx)
    adjnew = backend.zeros(nx)
    # Initialize CPML arrays
    ψ_l = backend.zeros(halo+1)
    ψ_r = backend.zeros(halo+1)
    ξ_l = backend.zeros(halo)
    ξ_r = backend.zeros(halo)
    ψ_adj_l = backend.zeros(halo+1)
    ψ_adj_r = backend.zeros(halo+1)
    ξ_adj_l = backend.zeros(halo)
    ξ_adj_r = backend.zeros(halo)
    # Wrap CPML coefficient arrays
    a_x_l = backend.Data.Array( model.cpmlcoeffs.a_l )
    a_x_r = backend.Data.Array( model.cpmlcoeffs.a_r )
    a_x_hl = backend.Data.Array( model.cpmlcoeffs.a_hl )
    a_x_hr = backend.Data.Array( model.cpmlcoeffs.a_hr )
    b_K_x_l = backend.Data.Array( model.cpmlcoeffs.b_K_l )
    b_K_x_r = backend.Data.Array( model.cpmlcoeffs.b_K_r )
    b_K_x_hl = backend.Data.Array( model.cpmlcoeffs.b_K_hl )
    b_K_x_hr = backend.Data.Array( model.cpmlcoeffs.b_K_hr )
    # Add empty row to source time functions (fix for last timestep)
    srctf = [srctf; zeros(1, size(srctf,2))]
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )
    observed_a = backend.Data.Array( observed )
    invcov_a = backend.Data.Array(invcov)
    # Checkpointing constants
    last_checkpoint = floor(Int, nt / check_freq) * check_freq   # time step of the last checkpoint
    # Checkpointing arrays
    save_buffer = backend.zeros(nx, check_freq + 2)
    checkpoints = Dict{Int, backend.Data.Array}()
    checkpoints_CPML = Dict{Int, Any}()
    checkpoints[-1] = copy(pold)
    checkpoints[0] = copy(pcur)
    checkpoints_CPML[0] = (copy(ψ_l), copy(ψ_r), copy(ξ_l), copy(ξ_r))
    # Residuals arrays
    residuals_a = similar(traces_a)
    # Gradient arrays
    curgrad = backend.zeros(nx)

    # Forward time loop (nt+1 timesteps)
    for it = 1:nt+1
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, dx,
            halo, ψ_l, ψ_r, ξ_l, ξ_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            possrcs_a, srctf_a, posrecs_a, traces_a, it;
            save_trace=(it <= nt)   # do not save trace for last timestep
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*it, maximum(abs.(Array(pcur))))
        end
        # Save checkpoint
        if it % check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current and last timestep pressure
            checkpoints[it] = copy(pcur)
            checkpoints[it-1] = copy(pold)
            # Also save CPML arrays
            checkpoints_CPML[it] = (copy(ψ_l), copy(ψ_r), copy(ξ_l), copy(ξ_r))
        end
        # Start populating save buffer just before last checkpoint
        if it >= last_checkpoint-1
            @debug @sprintf("Saving into buffer at iteration: %d", it)
            save_buffer[:, it - (last_checkpoint-1) + 1] .= pcur
        end
    end

    @debug "Computing residuals"
    # Compute residuals
    mul!(residuals_a, invcov_a, traces_a - observed_a)
    # Scale residuals
    residuals_a .*= model.dt^2

    @debug "Computing gradients"
    # Current checkpoint
    curr_checkpoint = last_checkpoint
    # Adjoint time loop (backward in time)
    for it = nt:-1:1
        # Compute one adjoint step
        adjold, adjcur, adjnew = backend.forward_onestep_CPML!(
            adjold, adjcur, adjnew, fact_a, dx,
            halo, ψ_adj_l, ψ_adj_r, ξ_adj_l, ξ_adj_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            posrecs_a, residuals_a, nothing, nothing, it;   # adjoint sources positions are receivers
            save_trace=false   # do not save traces for adjoint
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Backward iteration: %d, maximum absolute adjoint: %g", it, maximum(abs.(Array(adjcur))))
        end
        # Check if out of save buffer
        if it < curr_checkpoint
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            # Shift last checkpoint
            old_checkpoint = curr_checkpoint
            curr_checkpoint -= check_freq

            # Start forward computation from new last checkpoint

            # Shift start of save buffer
            save_buffer[:, 1] .= checkpoints[curr_checkpoint-1]
            save_buffer[:, 2] .= checkpoints[curr_checkpoint]
            # Shift end of save buffer
            save_buffer[:, end-1] .= checkpoints[old_checkpoint-1]
            save_buffer[:, end] .= checkpoints[old_checkpoint]
            # Recover arrays
            pold .= checkpoints[curr_checkpoint-1]
            pcur .= checkpoints[curr_checkpoint]
            pnew .= 0.0
            ψ_l .= checkpoints_CPML[curr_checkpoint][1]
            ψ_r .= checkpoints_CPML[curr_checkpoint][2]
            ξ_l .= checkpoints_CPML[curr_checkpoint][3]
            ξ_r .= checkpoints_CPML[curr_checkpoint][4]
            # Forward recovery time loop
            for recit = curr_checkpoint+1:(old_checkpoint-1)-1
                pold, pcur, pnew = backend.forward_onestep_CPML!(
                    pold, pcur, pnew, fact_a, dx,
                    halo, ψ_l, ψ_r, ξ_l, ξ_r,
                    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
                    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                    possrcs_a, srctf_a, nothing, nothing, recit;
                    save_trace=false   # do not save trace in recovery time loop
                )
                if recit % model.infoevery == 0
                    @debug @sprintf("Recovering iteration: %d, maximum absolute pressure: %g [Pa]", recit, maximum(abs.(Array(pcur))))
                end
                # Save recovered pressure in save buffer
                save_buffer[:, recit - (curr_checkpoint+1) + 3] .= pcur
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = save_buffer[:, it - curr_checkpoint + 1]
        pold_corr = save_buffer[:, it - curr_checkpoint + 2]
        pveryold_corr = save_buffer[:, it - curr_checkpoint + 3]
        # Correlate for gradient computation
        backend.correlate_gradient!(curgrad, adjcur, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end

    return Array( curgrad )
end