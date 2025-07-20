swgradient_1shot!(model::AcousticWaveSimulation, args...; kwargs...) =
    swgradient_1shot!(BoundaryConditionTrait(model), model, args...; kwargs...)

function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticCDWaveSimulation{T, N},
    shot::ScalarShot{T},
    misfit::AbstractMisfit
)::Dict{String, Array{T, N}} where {T, N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    # Get computational grid, checkpointer and backend
    grid = model.grid
    checkpointer = model.checkpointer
    backend = select_backend(typeof(model), model.runparams.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_bk = backend.Data.Array(possrcs)
    posrecs_bk = backend.Data.Array(posrecs)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Setup the print output 
    ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjforw)
        # Save checkpoint
        savecheckpoint!(checkpointer, "pcur" => grid.fields["pcur"], it)
        savecheckpoint!(checkpointer, "ψ" => grid.fields["ψ"], it)
        savecheckpoint!(checkpointer, "ξ" => grid.fields["ξ"], it)
    end

    @debug "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    adjointsource_bk = backend.Data.Array(.-∂χ_∂u(misfit, shot.recs))

    # Prescale residuals (fact = vel^2 * dt^2)
    backend.prescale_residuals!(adjointsource_bk, posrecs_bk, grid.fields["fact"].value)

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(model, posrecs_bk, adjointsource_bk, it)
        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjback)
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
                    backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, nothing, nothing, recit; save_trace=false)
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
    mutearoundmultiplepoints!(gradient,shot.srcs.positions,grid,model.smooth_radius)
    mutearoundmultiplepoints!(gradient,shot.recs.positions,grid,model.smooth_radius)
    # rescale gradient
    gradient .= (convert(T, 2.0) ./ (model.matprop.vp .^ 3)) .* gradient
    # # add regularization if needed
    # if misfit.regularization !== nothing
    #     gradient .+= dχ_dm(misfit.regularization, model.matprop)
    # end
    return Dict("vp" => gradient)
end

function swgradient_1shot!(
    ::CPMLBoundaryCondition,
    model::AcousticVDStaggeredCPMLWaveSimulation{T, N},
    shot::ScalarShot{T},
    misfit::AbstractMisfit
)::Dict{String, Array{T, N}} where {T, N}

    # scale source time function, etc.
    possrcs, posrecs, scal_srctf = possrcrec_scaletf(model, shot)

    # Get computational grid, checkpointer and backend
    grid = model.grid
    checkpointer = model.checkpointer
    backend = select_backend(typeof(model), model.runparams.parall)

    # Numerics
    nt = model.nt
    # Wrap sources and receivers arrays
    possrcs_bk = backend.Data.Array(possrcs)
    posrecs_bk = backend.Data.Array(posrecs)
    srctf_bk = backend.Data.Array(scal_srctf)
    traces_bk = backend.Data.Array(shot.recs.seismograms)
    # Reset wavesim
    reset!(model)

    # Setup the print output 
    ter = REPL.Terminals.TTYTerminal("", stdin, stdout, stderr)

    # Forward time loop
    for it in 1:nt
        # Compute one forward step
        backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, posrecs_bk, traces_bk, it)
        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjforw)
        # Save checkpoint
        savecheckpoint!(checkpointer, "pcur" => grid.fields["pcur"], it)
        savecheckpoint!(checkpointer, "vcur" => grid.fields["vcur"], it)
        savecheckpoint!(checkpointer, "ψ" => grid.fields["ψ"], it)
        savecheckpoint!(checkpointer, "ξ" => grid.fields["ξ"], it)
    end

    @debug "Saving seismograms"
    copyto!(shot.recs.seismograms, traces_bk)

    @debug "Computing residuals"
    adjointsource_bk = backend.Data.Array(.-∂χ_∂u(misfit, shot.recs))

    # Prescale residuals (fact = vel^2 * rho * dt)
    backend.prescale_residuals!(adjointsource_bk, posrecs_bk, grid.fields["fact_m0"].value)

    @debug "Computing gradients"
    # Adjoint time loop (backward in time)
    for it in nt:-1:1
        # Compute one adjoint step
        backend.adjoint_onestep_CPML!(model, posrecs_bk, adjointsource_bk, it)
        # Print timestep info
        printinfoiter(ter,it,nt,model.runparams.infoevery,model.dt,:adjback)
        # Check if out of save buffer
        if !issaved(checkpointer, "pcur", it - 1)
            @debug @sprintf("Out of save buffer at iteration: %d", it)
            initrecover!(checkpointer)
            copyto!(grid.fields["pcur"], getsaved(checkpointer, "pcur", checkpointer.curr_checkpoint))
            copyto!(grid.fields["vcur"], getsaved(checkpointer, "vcur", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ψ"], getsaved(checkpointer, "ψ", checkpointer.curr_checkpoint))
            copyto!(grid.fields["ξ"], getsaved(checkpointer, "ξ", checkpointer.curr_checkpoint))
            recover!(
                checkpointer,
                recit -> begin
                    backend.forward_onestep_CPML!(model, possrcs_bk, srctf_bk, nothing, nothing, recit; save_trace=false)
                    return ["pcur" => grid.fields["pcur"]]
                end
            )
        end
        # Get pressure fields from saved buffer
        pcur_corr = getsaved(checkpointer, "pcur", it).value
        pcur_old = getsaved(checkpointer, "pcur", it - 1).value
        # Correlate for gradient computation
        backend.correlate_gradient_m0!(grid.fields["grad_m0"].value, grid.fields["adjpcur"].value, pcur_corr, pcur_old, model.dt)
        backend.correlate_gradient_m1!(grid.fields["grad_m1_stag"].value, grid.fields["adjvcur"].value, pcur_corr, grid.spacing)
    end
    # Allocate gradients
    gradient_m0 = zeros(T, grid.size...)
    gradient_m1 = zeros(T, grid.size...)
    # Get gradients
    copyto!(gradient_m0, grid.fields["grad_m0"].value)
    for i in eachindex(grid.fields["grad_m1_stag"].value)
        # Accumulate and back interpolate staggered gradients
        gradient_m1 .+= back_interp(model.matprop.interp_method, 1 ./ model.matprop.rho, Array(grid.fields["grad_m1_stag"].value[i]), i)
    end
    # Smooth gradients
    mutearoundmultiplepoints!(gradient_m0,shot.srcs.positions,grid,model.smooth_radius)
    mutearoundmultiplepoints!(gradient_m1,shot.srcs.positions,grid,model.smooth_radius)
    mutearoundmultiplepoints!(gradient_m0,shot.recs.positions,grid,model.smooth_radius)
    mutearoundmultiplepoints!(gradient_m1,shot.recs.positions,grid,model.smooth_radius)

    # Rescale gradients with respect to material properties (chain rule)
    return Dict(
        "vp" => .-convert(T, 2.0) .* gradient_m0 ./ (model.matprop.vp .^ 3 .* model.matprop.rho), 
        "rho" => .-gradient_m0 ./ (model.matprop.vp .^ 2 .* model.matprop.rho .^ 2) .- gradient_m1 ./ model.matprop.rho # grad wrt rho
    )
end
