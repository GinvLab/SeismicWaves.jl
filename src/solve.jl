### UPDATE MATERIAL PROPERTIES ##

function set_wavesim_matprop!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}) where {N}
    @debug "Checking new material properties"
    check_matprop(wavesim, matprop)
    @debug "Updating WaveSimul material properties"
    update_matprop!(wavesim, matprop)
end

### FORWARDS ###

@views function run_swforward!(
    wavsim::WaveSimul{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
)::Union{Vector{Array}, Nothing} where {N}
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(wavsim, matprop, shots)

    # Set wavesim material properties
    # Remark: matprop in wavesim are mutated
    @info "Setting wavesim material properties"
    set_wavesim_matprop!(wavsim, matprop)
    # Now onwards matprop from outside should not be used anymore!!!

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
        swforward_1shot!(wavsim, possrcs, posrecs, srctf, traces)
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
    wavsim::WaveSimul{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
)::Real where {N}
    # Solve forward model for all shots
    run_swforward!(wavsim, matprop, shots)
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
    wavsim::WaveSimul{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    compute_misfit::Bool=false
)::Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}

    # Set wavesim material properties
    @info "Setting wavesim material properties"
    set_wavesim_matprop!(wavsim, matprop)

    # Initialize total gradient and total misfit
    totgrad = zeros(wavsim.totgrad_size...)
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
            wavsim, possrcs,
            posrecs, srctf, traces,
            recs.observed, recs.invcov
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
