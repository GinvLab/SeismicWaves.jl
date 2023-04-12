### FORWARDS ###

@views function run_swforward!(wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}}
)::Union{Vector{Array},
    Nothing}
    # Check wavsim
    @info "Checking wavsim"
    check(wavsim)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(wavsim)

    # Snapshots setup
    takesnapshots = snapenabled(wavsim)
    if takesnapshots
        snapshots_per_shot = []
    end

    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(wavsim, srcs, recs)
        # Compute forward solver
        @info "Forward modelling for one shot"
        swforward_1shot!(wavsim, backend, possrcs, posrecs, srctf, traces)
        # Save traces in reeivers seismograms
        @info "Saving seismograms"
        copyto!(recs.seismograms, traces)
        # Save shot's snapshots
        if takesnapshots
            @info "Saving snapshots"
            push!(snapshots_per_shot, copy(wavsim.snapshots))
        end
    end

    if takesnapshots
        return snapshots_per_shot
    end
    return nothing
end

### MISFITS ###

@views function run_swmisfit!(wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}})::Real
    # Solve forward model for all shots
    run_swforward!(wavsim, backend, shots)
    # Compute total misfit for all shots
    misfit = 0
    for (shot, (_, recs)) in enumerate(shots)
        @info "Computing residuals for shot #$(shot)"
        residuals = similar(recs.seismograms)
        @info "Checking invcov matrix for shot #$(shot)"
        check_invcov_matrix(wavsim, recs.invcov)
        @info "Computing misfit for shot #$(shot)"
        difcalobs = recs.seismograms - recs.observed
        mul!(residuals, recs.invcov, difcalobs)
        misfit += dot(difcalobs, residuals)
    end

    return misfit / 2
end

### GRADIENTS ###

@views function run_swgradient!(wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    check_freq::Union{Integer, Nothing}=nothing)::AbstractArray
    # Check wavsim
    @info "Checking wavsim"
    check(wavsim)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(wavsim)
    # Check checkpointing setup
    @info "Checking checkpointing frequency"
    check_checkpoint_frequency(wavsim, check_freq)

    # Initialize total gradient
    totgrad = zero(wavsim.vel)

    # Shots loop
    for (shot, (srcs, recs)) in enumerate(shots)
        @info "Shot #$(shot)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(wavsim, srcs, recs)
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad =
            swgradient_1shot!(wavsim, backend, possrcs, posrecs, srctf, traces, recs.observed, recs.invcov; check_freq=check_freq)
        # Accumulate gradient
        totgrad .+= curgrad
    end

    return totgrad
end
