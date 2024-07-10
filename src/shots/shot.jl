
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
