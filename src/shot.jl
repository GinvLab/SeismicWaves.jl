
"""
$(TYPEDEF)

Type representing a source-receiver pair, i.e., a \"shot\".

$(TYPEDFIELDS)
"""
Base.@kwdef struct Shot
    "Structure containing the appropriate Sources for a given simulation."
    srcs::Sources
    "Structure containing the appropriate Receivers for a given simulation."
    recs::Receivers
end

##################################################

function init_shot!(model::WaveSimul, shot::Shot; kwargs...)
    # Check shot configuration
    check_shot(model, shot; kwargs...)
    # Initialize boundary conditions based on current shot
    init_bdc!(model, shot.srcs)
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
