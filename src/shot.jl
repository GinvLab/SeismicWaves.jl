
"""
$(TYPEDEF)

Type representing a shot with scalar sources and receivers.

$(TYPEDFIELDS)
"""
Base.@kwdef struct ScalarShot{T} <: Shot{T}
    "Structure containing the ScalarSources for a given simulation."
    srcs::ScalarSources{T}
    "Structure containing the ScalarReceivers for a given simulation."
    recs::ScalarReceivers{T}
end

"""
$(TYPEDEF)

Type representing a shot with moment tensor sources and multi-component receivers.

$(TYPEDFIELDS)
"""
Base.@kwdef struct MomentTensorShot{T, N, M <: MomentTensor{T, N}} <: Shot{T}
    "Structure containing the MomentTensorSources for a given simulation."
    srcs::MomentTensorSources{T, N, M}
    "Structure containing the VectorReceivers for a given simulation."
    recs::VectorReceivers{T, N}
end



##################################################

function init_shot!(model::WaveSimulation{T}, shot::Shot{T}; kwargs...) where {T}
    # Check shot configuration
    check_shot(model, shot; kwargs...)
    # Initialize boundary conditions based on current shot
    init_bdc!(model, shot.srcs)
end

@views function find_nearest_grid_points(model::WaveSimulation{T}, positions::Matrix{T})::Matrix{Int} where {T}
    # source time functions
    nsrcs = size(positions, 1)                      # number of sources
    ncoos = size(positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    idx_positions = zeros(Int, size(positions))     # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [positions[s, i] / model.grid.spacing[i] + 1 for i in 1:ncoos]
        idx_positions[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    return idx_positions
end