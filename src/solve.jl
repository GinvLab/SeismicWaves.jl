### UPDATE MATERIAL PROPERTIES ##

@views function set_wavesim_matprop!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}) where {N}
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

    # Check wavesim consistency
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
    #Threads.@threads for singleshot in shots
    parall = wavsim.parall
    @athreadpersrcornot for singleshot in shots
        srcs = singleshot.srcs
        recs = singleshot.recs
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf = init_shot!(wavsim, singleshot)
        # Compute forward solver
        @info "Forward modelling for one shot"
        swforward_1shot!(wavsim, possrcs, posrecs, srctf, recs)
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
    misfit::AbstractMisfit=L2Misfit(nothing)
)::Real where {N}
    # Solve forward model for all shots
    run_swforward!(wavsim, matprop, shots)
    # Compute total misfit for all shots
    totmisfit = 0
    @athreadpersrcornot for singleshot in shots
        srcs = singleshot.srcs
        recs = singleshot.recs
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, recs.invcov)
        @info "Computing misfit"
        totmisfit += misfit(recs, matprop)
    end

    return totmisfit
end

### GRADIENTS ###

@views function run_swgradient!(
    wavsim::WaveSimul{N},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing)
)::Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}

    # Check wavesim consistency
    @debug "Checking consistency across simulation type, material parameters and source-receiver types"
    check_sim_consistency(wavsim, matprop, shots)

    # Set wavesim material properties
    @info "Setting wavesim material properties"
    set_wavesim_matprop!(wavsim, matprop)

    # Initialize total gradient and total misfit
    parall = wavsim.parall
    if parall==:athreadpersrc
        allgrad = [zeros(wavsim.totgrad_size...) for i=1:length(shots)]
    else
        totgrad = zeros(wavsim.totgrad_size...)        
    end
    totmisfit = 0
    # Shots loop
    s = 0
    @athreadpersrcornot for singleshot in shots
        s+=1
        @info "Shot #$s"
        srcs = singleshot.srcs
        recs = singleshot.recs
        # Initialize shot
        @info "Initializing shot"
        possrcs, posrecs, srctf = init_shot!(wavsim, singleshot)
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        curgrad = swgradient_1shot!(
            wavsim, possrcs,
            posrecs, srctf,
            recs, misfit
        )
        # Compute misfit if needed
        if compute_misfit
            @info "Computing misfit"
            totmisfit += misfit(recs, matprop)
        end
        # Accumulate gradient
        if parall==:athreadpersrc
            allgrad[s] = curgrad
        else
            totgrad .+= curgrad
        end
    end
    if parall==:athreadpersrc
        totgrad = sum(allgrad)
    end

    return compute_misfit ? (totgrad, totmisfit) : totgrad
end
