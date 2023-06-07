
"""
Type representing a source-receiver pair, i.e., a \"shot\".
"""
Base.@kwdef struct Shot #{T <: Real}
    srcs::Sources #{T}
    recs::Receivers #{T}
end

##################################################

function init_shot!(model::WaveSimul, shot::Shot; kwargs...)::Tuple{Matrix{<:Int}, Matrix{<:Int}, Array{<:Real}, Array{<:Real}}
    # Check shot configuration
    check_shot(model, shot; kwargs...)
    # Initialize boundary conditions based on current shot
    init_bdc!(model, shot.srcs)
    # Return allocated shot's arrays
    return setup_shot(model, shot)
end

@views function setup_shot(model::WaveSimul, shot::Shot)::Tuple{Matrix{<:Int}, Matrix{<:Int}, Array{<:Real}, Array{<:Real}}
    # find nearest grid points indexes for both sources and receivers
    possrcs = find_nearest_grid_points(model, shot.srcs.positions)
    posrecs = find_nearest_grid_points(model, shot.recs.positions)
    # source time function (prescale if needed)
    scaled_tf = scale_srctf(model, shot.srcs.tf, possrcs)
    # initialize traces array (same shape as receivers seismograms)
    traces = zero(shot.recs.seismograms)

    return possrcs, posrecs, scaled_tf, traces
end

@views function find_nearest_grid_points(model::WaveSimul, positions::Matrix{<:Real})::Matrix{<:Int}
    # source time functions
    nsrcs = size(positions, 1)                      # number of sources
    ncoos = size(positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    idx_positions = zeros(Int, size(positions))     # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [positions[s, i] / model.gridspacing[i] + 1 for i in 1:ncoos]
        idx_positions[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    return idx_positions
end
