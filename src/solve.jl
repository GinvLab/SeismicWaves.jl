### FORWARDS ###

@views function forward!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    backend
    )::Union{Vector{Array}, Nothing}
    # Check model
    @info "Checking model"
    check(model)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(model)

    # Snapshots setup
    if snapenabled(model)
        snapshots_per_shot = []
    end
    
    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        # Compute forward solver
        @info "Computing forward solver"
        forward!(model, possrcs, posrecs, srctf, traces, backend)
        # Save traces in reeivers seismograms
        @info "Saving receivers seismograms"
        copyto!(recs.seismograms, traces)
        # Save shot's snapshots
        if snapenabled(model)
            @info "Saving snapshots"
            push!(snapshots_per_shot, copy(model.snapshots))
        end
    end

    if snapenabled(model)
        return snapshots_per_shot
    end
    return nothing
end

forward!(model::WaveModel, possrcs, posrecs, srctf, traces, backend) = forward!(
    WaveEquationTrait(model),
    BoundaryConditionTrait(model),
    model, possrcs, posrecs, srctf, traces, backend
)

### MISFITS ###

@views function misfit!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    backend
    )::Real
    # Solve forward model for all shots
    forward!(model, shots, backend)
    # Compute total misfit for all shots
    misfit = 0
    for (shot, (_, recs)) in enumerate(shots)
        @info "Computing residuals for shot #$(shot)"
        residuals = similar(recs.seismograms)
        @info "Checking invcov matrix for shot #$(shot)"
        check_invcov_matrix(model, recs.invcov)
        @info "Computing misfit for shot #$(shot)"
        difcalobs = recs.seismograms - recs.observed
        mul!(residuals, recs.invcov, difcalobs)
        misfit += dot(difcalobs, residuals)
    end

    return misfit / 2
end

### GRADIENTS ###

@views function gradient!(
    model::WaveModel,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}},
    backend;
    check_freq::Union{Integer, Nothing} = nothing
    )::AbstractArray
    # Check model
    @info "Checking model"
    check(model)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(model)
    # Check checkpointing setup
    @info "Checking checkpointing frequency"
    check_checkpoint_frequency(model, check_freq)

    # Initialize total gradient
    totgrad = zero(model.vel)
    
    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(model, srcs, recs)
        @info "Checking invcov matrix"
        check_invcov_matrix(model, recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad = gradient!(model, possrcs, posrecs, srctf, traces, recs.observed, recs.invcov, backend; check_freq=check_freq)
        # Accumulate gradient
        totgrad .+= curgrad
    end
    
    return totgrad
end

gradient!(model::WaveModel, possrcs, posrecs, srctf, traces, observed, invcov, backend; check_freq=check_freq) = gradient!(
    WaveEquationTrait(model),
    BoundaryConditionTrait(model),
    model, possrcs, posrecs, srctf, traces, observed, invcov, backend; check_freq=check_freq
)
