"""
$(TYPEDEF)

Type representing a multi-source configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarSources{T} <: Sources{T}
    "Source positions"
    positions::Matrix{T}
    "Source time function"
    tf::Matrix{T}
    "Dominant frequency"
    domfreq::T

    @doc """ 
        ScalarSources(positions::Matrix{T}, tf::Matrix{T}, domfreq::T) where {T}

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources(positions::Matrix{T}, tf::Matrix{T}, domfreq::T) where {T}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new{T}(positions, tf, domfreq)
    end
end

####################################################

"""
Type representing components of a 2D moment tensor.
"""
Base.@kwdef struct MomentTensor2D{T} <: MomentTensor{T, 2}
    Mxx::T
    Mzz::T
    Mxz::T
end

"""
Type representing components of a 3D moment tensor.
"""
Base.@kwdef struct MomentTensor3D{T} <: MomentTensor{T, 3}
    Mxx::T
    Myy::T
    Mzz::T
    Mxy::T
    Mxz::T
    Myz::T
end

"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct MomentTensorSources{T, N, M <: MomentTensor{T, N}} <: Sources{T}
    positions::Matrix{T}
    tf::Matrix{T}
    momtens::Vector{M}
    domfreq::T

    @doc """
        MomentTensorSources( 
            positions::Matrix{T},
            tf::Matrix{T},
            momtens::Vector{M}
            domfreq::T
        ) where {T, N, M <: MomentTensor{T}}

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function MomentTensorSources(positions::Matrix{T}, tf::Matrix{T}, momtens::Vector{M}, domfreq::T) where {T, N, M <: MomentTensor{T, N}}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        @assert length(momtens) == size(positions, 1) "Number of moment tensors must match number of sources!"
        return new{T, N, M}(positions, tf, momtens, domfreq)
    end
end

struct ExternalForceSources{T, N} <: Sources{T}
    positions::Matrix{T}
    tf::Array{T, 3}
    domfreq::T

    function ExternalForceSources(positions::Matrix{T}, tf::Array{T, 3}, domfreq::T) where {T}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 3) "Number of sources do not match between positions and time-functions!"
        @assert size(tf, 2) == size(positions, 2) "Number of components do not match between time-functions and positions!"
        N = size(tf, 2)
        return new{T, N}(positions, tf, domfreq)
    end
end

struct PSDMomentTensorSources{T, N, M <: MomentTensor{T, N}} <: Sources{T}
    positions::Matrix{T}
    tf::Array{T, 3}
    domfreq::T
    psd::Vector{M}

    function PSDMomentTensorSources(positions::Matrix{T}, tf::Array{T, 3}, domfreq::T, psd::Vector{M}) where {T, N, M <: MomentTensor{T, N}}
        @assert size(positions, 1) > 0 "There must be at least one PSD source!"
        @assert size(positions, 1) == length(psd) "Number of PSD sources must match number of positions!"
        @assert size(positions, 2) == N "Number of components in positions must match the number of dimensions!"
        @assert size(tf, 3) == 1 "PSD moment tensor sources must have a single source time function!"
        @assert size(tf, 2) == N "Number of components in source time function must match the number of dimensions!"
        return new{T, N, M}(positions, tf, domfreq, psd)
    end 
end

####################################################
