@views function gradient!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, observed, invcov, backend;
    check_freq=nothing
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
    # Initialize adjoint CPML arrays
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
    invcov_a = backend.Data.Array( invcov )

    # Checkpointing setup
    if check_freq !== nothing   # checkpointing enabled
        # Time step of last checkpoint
        last_checkpoint = floor(Int, nt / check_freq) * check_freq
        # Checkpointing arrays
        save_buffer = backend.zeros(nx, check_freq + 2)     # pressure window buffer
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints
        checkpoints_CPML = Dict{Int, Any}()                 # CPML arrays checkpoints
        # Save initial conditions as first checkpoint
        checkpoints[-1] = copy(pold)
        checkpoints[0] = copy(pcur)
        checkpoints_CPML[0] = (copy(ψ_l), copy(ψ_r), copy(ξ_l), copy(ξ_r))
    else    # no checkpointing
        last_checkpoint = 1                                 # simulate a checkpoint at time step 1 (so buffer will start from 0)
        save_buffer = backend.zeros(nx, nt+2)               # save all timesteps (from 0 to n+1 so n+2)
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
        checkpoints_CPML = Dict{Int, Any}()                 # CPML arrays checkpoints (will remain empty)
        # Save time step 0 into buffer
        save_buffer[:, 1] .= pcur
    end
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
            @debug @sprintf("Forward iteration: %d, simulation time: %g [s]", it, model.dt*it)
        end
        # Save checkpoint
        if check_freq !== nothing && it % check_freq == 0
            @debug @sprintf("Saving checkpoint at iteration: %d", it)
            # Save current and last timestep pressure
            checkpoints[it] = copy(pcur)
            checkpoints[it-1] = copy(pold)
            # Also save CPML arrays
            checkpoints_CPML[it] = [copy(ψ_l), copy(ψ_r), copy(ξ_l), copy(ξ_r)]
        end
        # Start populating save buffer just before last checkpoint
        if it >= last_checkpoint-1
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
            save_buffer[:, 1] .= checkpoints[curr_checkpoint-1]
            save_buffer[:, 2] .= checkpoints[curr_checkpoint]
            # Shift end of save buffer
            save_buffer[:, end-1] .= checkpoints[old_checkpoint-1]
            save_buffer[:, end] .= checkpoints[old_checkpoint]
            # Recover pressure and CPML arrays from current checkpoint
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
                    @debug @sprintf("Recovering iteration: %d", recit)
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

@views function gradient!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel2D, possrcs, posrecs, srctf, traces, observed, invcov, backend;
    check_freq=nothing
)
    # Numerics
    nt = model.nt
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx, ny)
    pcur = backend.zeros(nx, ny)
    pnew = backend.zeros(nx, ny)
    fact_a = backend.Data.Array( model.fact )
    # Initialize adjoint arrays
    adjold = backend.zeros(nx, ny)
    adjcur = backend.zeros(nx, ny)
    adjnew = backend.zeros(nx, ny)
    # Initialize CPML arrays
    ψ_x_l = backend.zeros(halo+1, ny)
    ψ_x_r = backend.zeros(halo+1, ny)
    ξ_x_l = backend.zeros(halo, ny)
    ξ_x_r = backend.zeros(halo, ny)
    ψ_y_l = backend.zeros(nx, halo+1)
    ψ_y_r = backend.zeros(nx, halo+1)
    ξ_y_l = backend.zeros(nx, halo)
    ξ_y_r = backend.zeros(nx, halo)
    # Initialize adjoint CPML arrays
    ψ_adj_x_l = backend.zeros(halo+1, ny)
    ψ_adj_x_r = backend.zeros(halo+1, ny)
    ξ_adj_x_l = backend.zeros(halo, ny)
    ξ_adj_x_r = backend.zeros(halo, ny)
    ψ_adj_y_l = backend.zeros(nx, halo+1)
    ψ_adj_y_r = backend.zeros(nx, halo+1)
    ξ_adj_y_l = backend.zeros(nx, halo)
    ξ_adj_y_r = backend.zeros(nx, halo)
    # Wrap CPML coefficient arrays
    a_x_l = backend.Data.Array( model.cpmlcoeffs_x.a_l )
    a_x_r = backend.Data.Array( model.cpmlcoeffs_x.a_r )
    a_x_hl = backend.Data.Array( model.cpmlcoeffs_x.a_hl )
    a_x_hr = backend.Data.Array( model.cpmlcoeffs_x.a_hr )
    b_K_x_l = backend.Data.Array( model.cpmlcoeffs_x.b_K_l )
    b_K_x_r = backend.Data.Array( model.cpmlcoeffs_x.b_K_r )
    b_K_x_hl = backend.Data.Array( model.cpmlcoeffs_x.b_K_hl )
    b_K_x_hr = backend.Data.Array( model.cpmlcoeffs_x.b_K_hr )
    a_y_l = backend.Data.Array( model.cpmlcoeffs_y.a_l )
    a_y_r = backend.Data.Array( model.cpmlcoeffs_y.a_r )
    a_y_hl = backend.Data.Array( model.cpmlcoeffs_y.a_hl )
    a_y_hr = backend.Data.Array( model.cpmlcoeffs_y.a_hr )
    b_K_y_l = backend.Data.Array( model.cpmlcoeffs_y.b_K_l )
    b_K_y_r = backend.Data.Array( model.cpmlcoeffs_y.b_K_r )
    b_K_y_hl = backend.Data.Array( model.cpmlcoeffs_y.b_K_hl )
    b_K_y_hr = backend.Data.Array( model.cpmlcoeffs_y.b_K_hr )
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
        save_buffer = backend.zeros(nx, ny, check_freq + 2)     # pressure window buffer
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints
        checkpoints_CPML = Dict{Int, Any}()                 # CPML arrays checkpoints
        # Save initial conditions as first checkpoint
        checkpoints[-1] = copy(pold)
        checkpoints[0] = copy(pcur)
        checkpoints_CPML[0] = [
            copy(ψ_x_l), copy(ψ_x_r), copy(ξ_x_l), copy(ξ_x_r),
            copy(ψ_y_l), copy(ψ_y_r), copy(ξ_y_l), copy(ξ_y_r),
        ]
    else    # no checkpointing
        last_checkpoint = 1                                 # simulate a checkpoint at time step 1 (so buffer will start from 0)
        save_buffer = backend.zeros(nx, ny, nt+2)           # save all timesteps (from 0 to n+1 so n+2)
        checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
        checkpoints_CPML = Dict{Int, Any}()                 # CPML arrays checkpoints (will remain empty)
        # Save time step 0 into buffer
        save_buffer[:, :, 1] .= pcur
    end
    # Residuals arrays
    residuals_a = similar(traces_a)
    # Gradient arrays
    curgrad = backend.zeros(nx, ny)

    # Forward time loop (nt+1 timesteps)
    for it = 1:nt+1
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, dx, dy,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
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
            checkpoints_CPML[it] = [
                copy(ψ_x_l), copy(ψ_x_r), copy(ξ_x_l), copy(ξ_x_r),
                copy(ψ_y_l), copy(ψ_y_r), copy(ξ_y_l), copy(ξ_y_r),
            ]
        end
        # Start populating save buffer just before last checkpoint
        if it >= last_checkpoint-1
            save_buffer[:, :, it - (last_checkpoint-1) + 1] .= pcur
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
            adjold, adjcur, adjnew, fact_a, dx, dy,
            halo, ψ_adj_x_l, ψ_adj_x_r, ξ_adj_x_l, ξ_adj_x_r, ψ_adj_y_l, ψ_adj_y_r, ξ_adj_y_l, ξ_adj_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
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
            save_buffer[:, :, 1] .= checkpoints[curr_checkpoint-1]
            save_buffer[:, :, 2] .= checkpoints[curr_checkpoint]
            # Shift end of save buffer
            save_buffer[:, :, end-1] .= checkpoints[old_checkpoint-1]
            save_buffer[:, :, end] .= checkpoints[old_checkpoint]
            # Recover pressure and CPML arrays from current checkpoint
            pold .= checkpoints[curr_checkpoint-1]
            pcur .= checkpoints[curr_checkpoint]
            pnew .= 0.0
            ψ_x_l .= checkpoints_CPML[curr_checkpoint][1]
            ψ_x_r .= checkpoints_CPML[curr_checkpoint][2]
            ξ_x_l .= checkpoints_CPML[curr_checkpoint][3]
            ξ_x_r .= checkpoints_CPML[curr_checkpoint][4]
            ψ_y_l .= checkpoints_CPML[curr_checkpoint][5]
            ψ_y_r .= checkpoints_CPML[curr_checkpoint][6]
            ξ_y_l .= checkpoints_CPML[curr_checkpoint][7]
            ξ_y_r .= checkpoints_CPML[curr_checkpoint][8]
            # Forward recovery time loop
            for recit = curr_checkpoint+1:(old_checkpoint-1)-1
                pold, pcur, pnew = backend.forward_onestep_CPML!(
                    pold, pcur, pnew, fact_a, dx, dy,
                    halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
                    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
                    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
                    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
                    possrcs_a, srctf_a, nothing, nothing, recit;
                    save_trace=false   # do not save trace in recovery time loop
                )
                if recit % model.infoevery == 0
                    @debug @sprintf("Recovering iteration: %d", recit)
                end
                # Save recovered pressure in save buffer
                save_buffer[:, :, recit - (curr_checkpoint+1) + 3] .= pcur
            end
        end
        # Get pressure fields from saved buffer
        pcur_corr = save_buffer[:, :, it - curr_checkpoint + 1]
        pold_corr = save_buffer[:, :, it - curr_checkpoint + 2]
        pveryold_corr = save_buffer[:, :, it - curr_checkpoint + 3]
        # Correlate for gradient computation
        backend.correlate_gradient!(curgrad, adjcur, pcur_corr, pold_corr, pveryold_corr, model.dt)
    end

    return Array( curgrad )
end

