### FORWARDS ###

@views function run_swforward!(
    wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Shot} ; #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}}
)::Union{Vector{Array}, Nothing}
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
    for (s, singleshot) in enumerate(shots)
        srcs = singleshot.srcs
        recs = singleshot.recs
        @info "Shot #$(s)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(wavsim, singleshot)
        # Compute forward solver
        @info "Forward modelling for one shot"
        swforward_1shot!(wavsim, backend, possrcs, posrecs, srctf, traces)
        # Save traces in receivers seismograms
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

@views function run_swmisfit!(
    wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Shot} ; #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}}
)::Real
    # Solve forward model for all shots
    run_swforward!(wavsim, backend, shots)
    # Compute total misfit for all shots
    totmisfit = 0
    for (s, singleshot) in enumerate(shots)
        srcs = singleshot.srcs
        recs = singleshot.recs
        @info "Shot #$(s)"
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, recs.invcov)
        @info "Computing misfit"
        residuals = similar(recs.seismograms)
        difcalobs = recs.seismograms - recs.observed
        mul!(residuals, recs.invcov, difcalobs)
        totmisfit += dot(difcalobs, residuals)
    end

    return totmisfit / 2
end

### GRADIENTS ###

@views function run_swgradient!(
    wavsim::WaveSimul,
    backend::Module,
    shots::Vector{<:Shot} ; #<:Pair{<:Sources{<:Real}, <:Receivers{<:Real}}};
    check_freq::Union{Integer, Nothing}=nothing,
    compute_misfit::Bool=false
)::Union{AbstractArray, Tuple{AbstractArray, Real}}
    # Check wavsim
    @info "Checking wavsim"
    check(wavsim)
    # Precompute constant values
    @info "Precomputing constant values"
    precompute!(wavsim)
    # Check checkpointing setup
    @info "Checking checkpointing frequency"
    check_checkpoint_frequency(wavsim, check_freq)

    # Initialize total gradient and total misfit
    totgrad = zero(wavsim.vel)
    totmisfit = 0
    # Shots loop
    for (s, singleshot) in enumerate(shots)
        srcs = singleshot.srcs
        recs = singleshot.recs
        @info "Shot #$(s)"
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf, traces = init_shot!(wavsim, singleshot)
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad = swgradient_1shot!(
            wavsim, backend, possrcs,
            posrecs, srctf, traces,
            recs.observed, recs.invcov;
            check_freq=check_freq
        )
        # Compute misfit
        @info "Saving seismograms"
        copyto!(recs.seismograms, traces)
        # Compute misfit if needed
        if compute_misfit
            @info "Computing misfit"
            residuals = similar(recs.seismograms)
            difcalobs = recs.seismograms - recs.observed
            mul!(residuals, recs.invcov, difcalobs)
            totmisfit += dot(difcalobs, residuals)
        end
        # Accumulate gradient
        totgrad .+= curgrad
    end

    return compute_misfit ? (totgrad, totmisfit / 2) : totgrad
end
