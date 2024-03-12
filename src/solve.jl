### UPDATE MATERIAL PROPERTIES ##

@views function set_wavesim_matprop!(wavesim::WaveSimul{N}, matprop::MaterialProperties{N}) where {N}
    @debug "Checking new material properties"
    check_matprop(wavesim, matprop)
    @debug "Updating WaveSimul material properties"
    update_matprop!(wavesim, matprop)
end

### FORWARDS ###


## single WaveSimul object
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
    # Now onwards matprop from "outside" should not be used anymore!!!

    # Snapshots setup
    takesnapshots = snapenabled(wavsim)
    if takesnapshots
        snapshots_per_shot = []
    end

    # Shots loop
    for singleshot in shots
        # Initialize shot
        @info "Initializing shot"
        #possrcs, posrecs, srctf = init_shot!(wavsim, singleshot)
        init_shot!(wavsim, singleshot)
        # Compute forward solver
        @info "Forward modelling for one shot"
        swforward_1shot!(wavsim, singleshot) #possrcs, posrecs, srctf, recs)
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

## :threadpersrc, multiple WaveSimul objects
@views function run_swforward!(
    wavsim::Vector{<:WaveSimul{N}},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    )::Union{Vector{Array}, Nothing} where {N}

    nwsim = length(wavsim)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert nthr == nwsim

    takesnapshots = false
    for w=1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(wavsim[w], matprop, shots)
        # Set wavesim material properties
        # Remark: matprop in wavesim are mutated
        @info "Setting wavesim material properties"
        set_wavesim_matprop!(wavsim[w], matprop)
        # Now onwards matprop from outside should not be used anymore!!!
  
        # Snapshots setup
        takesnapshots = snapenabled(wavsim[w])
        if takesnapshots
            snapshots_per_shot = [[] for j=1:nwsim]
        end
    end

    # Shots loop
    nshots = length(shots)
    grpshots = distribsrcs(nshots,nthr) # a vector of UnitRange 
    # loop on the set of WaveSimul
    Threads.@threads for w=1:nwsim
        # loop on the subset of shots per each WaveSimul 
        for s in grpshots[w]  
            singleshot = shots[s]
            # Initialize shot
            @info "Initializing shot"
            #possrcs, posrecs, srctf = init_shot!(wavsim[w], singleshot)
            init_shot!(wavsim[w], singleshot)
            # Compute forward solver
            @info "Forward modelling for one shot"
            #swforward_1shot!(wavsim[w], possrcs, posrecs, srctf, recs)
            swforward_1shot!(wavsim[w], singleshot)
            # Save shot's snapshots
            if takesnapshots
                @info "Saving snapshots"
                push!(snapshots_per_shot[w], copy(wavsim[w].snapshots))
            end
        end
    end

    if takesnapshots
        return collect(snapshots_per_shot)
    end
    return nothing
end


### MISFITS ###

## single or multiple WaveSimul objects
@views function run_swmisfit!(
    wavsim::Union{WaveSimul{N},Vector{<:WaveSimul{N}}},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    misfit::AbstractMisfit=L2Misfit(nothing)
    )::Real where {N}
    
    # Solve forward model for all shots
    run_swforward!(wavsim, matprop, shots)
    # Compute total misfit for all shots
    totmisfitval = 0
    for singleshot in shots
        @info "Checking invcov matrix"
        if typeof(wavsim)<:Vector{<:WaveSimul} 
            check_invcov_matrix(wavsim[1], singleshot.recs.invcov)
        else
            check_invcov_matrix(wavsim, singleshot.recs.invcov)
        end
        @info "Computing misfit"
        totmisfitval += misfit(singleshot.recs, matprop)
    end

    return totmisfitval
end


### GRADIENTS ###

## single WaveSimul object
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
    totgrad = zeros(wavsim.totgrad_size...)
    totmisfitval = 0
    # Shots loop
    for (s,singleshot) in enumerate(shots)
        @info "Shot #$s"
        # Initialize shot
        @info "Initializing shot"
        #possrcs, posrecs, srctf = init_shot!(wavsim, singleshot)
        init_shot!(wavsim, singleshot)
        @info "Checking invcov matrix"
        check_invcov_matrix(wavsim, singleshot.recs.invcov)
        # Compute forward solver
        @info "Computing gradient solver"
        # curgrad = swgradient_1shot!(
        #     wavsim, possrcs,
        #     posrecs, srctf,
        #     recs, misfit
        # )
        curgrad = swgradient_1shot!(
            wavsim, singleshot, misfit
        )
        # Accumulate gradient
        totgrad .+= curgrad
        # Compute misfit if needed
        if compute_misfit
            @info "Computing misfit"
            totmisfitval += misfit(singleshot.recs, matprop)
        end
    end

    return compute_misfit ? (totgrad, totmisfitval) : totgrad
end


## :threadpersrc, multiple WaveSimul objects
@views function run_swgradient!(
    wavsim::Vector{<:WaveSimul{N}},
    matprop::MaterialProperties{N},
    shots::Vector{<:Shot};
    compute_misfit::Bool=false,
    misfit::AbstractMisfit=L2Misfit(nothing)
    )::Union{AbstractArray, Tuple{AbstractArray, Real}} where {N}

    nwsim = length(wavsim)
    nthr = Threads.nthreads()
    # make sure the number of threads has not changed!
    @assert Threads.nthreads() == nwsim

    for w=1:nwsim
        # Check wavesim consistency
        @debug "Checking consistency across simulation type, material parameters and source-receiver types"
        check_sim_consistency(wavsim[w], matprop, shots)

        # Set wavesim material properties
        @info "Setting wavesim material properties"
        set_wavesim_matprop!(wavsim[w], matprop)
    end

    # Initialize total gradient and total misfit
    nshots = length(shots)
    allgrad = [zeros(wavsim[1].totgrad_size...) for i=1:nshots]

    if compute_misfit
        allmisfitval = zeros(nshots)
        totmisfitval = 0
    end

    # Shots loop
    nshots = length(shots)
    grpshots = distribsrcs(nshots,nthr) # a vector of UnitRange 
    # loop on the set of WaveSimul
    Threads.@threads for w=1:nwsim
        # loop on the subset of shots per each WaveSimul 
        for s in grpshots[w]  
            singleshot = shots[s]
            @info "Shot #$s"
            # Initialize shot
            @info "Initializing shot"
            #possrcs, posrecs, srctf = init_shot!(wavsim[w], singleshot)
            init_shot!(wavsim[w], singleshot)
            @info "Checking invcov matrix"
            check_invcov_matrix(wavsim[w], singleshot.recs.invcov)
            # Compute forward solver
            @info "Computing gradient solver"
            # curgrad = swgradient_1shot!(
            #     wavsim[w], possrcs,
            #     posrecs, srctf,
            #     recs, misfit
            # )
            curgrad = swgradient_1shot!(
                wavsim[w], singleshot, misfit
            )
            # Accumulate gradient
            allgrad[s] = curgrad
            # Compute misfit if needed
            if compute_misfit
                @info "Computing misfit"
                allmisfitval[s] = misfit(singleshot.recs, matprop)
            end
        end
    end
    
    totgrad = sum(allgrad)
    if compute_misfit
        totmisfitval = sum(allmisfitval)
    end
    
    return compute_misfit ? (totgrad, totmisfitval) : totgrad
end
