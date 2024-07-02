"""
$(TYPEDEF)

Type representing a multi-source configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarSources{T} <: Sources
    "Source positions"
    positions::Matrix{T}
    "Source time function"
    tf::Matrix{T}
    "Dominant frequency"
    domfreq::T

    @doc """ 
        ScalarSources{T}(positions::Matrix{T}, tf::Matrix{T}, domfreq::T) where {T}

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources{T}(positions::Matrix{T}, tf::Matrix{T}, domfreq::T) where {T}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new(positions, tf, domfreq)
    end
end

# Default type constructor {Float64}
@doc """
$(SIGNATURES)

Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
Default type constructor for Float64.
"""
ScalarSources(positions, tf, domfreq) = ScalarSources{Float64}(positions, tf, domfreq)

####################################################

"""
Type representing vector components of a 2D moment tensor.
"""
Base.@kwdef struct MomentTensor2D{T} <: MomentTensor{T}
    Mxx::Vector{T}
    Mzz::Vector{T}
    Mxz::Vector{T}
end

"""
Type representing vector components of a 2D moment tensor.
"""
Base.@kwdef struct MomentTensor3D{T} <: MomentTensor{T}
    Mxx::Vector{T}
    Myy::Vector{T}
    Mzz::Vector{T}
    Mxy::Vector{T}
    Mxz::Vector{T}
    Myz::Vector{T}
end

"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct MomentTensorSources{N, T} <: Sources
    positions::Matrix{T}
    tf::Matrix{T}
    momtens::MomentTensor{T}
    domfreq::T

    @doc """
        MomentTensorSources{T}( 
            positions::Matrix{T},
            tf::Matrix{T},
            momtens::MomentTensor
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function MomentTensorSources{N, T}(positions::Matrix{T}, tf::Matrix{T}, momtens::MomentTensor{T}, domfreq::T) where {N, T}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        if N == 2
            @assert typeof(momtens) <: MomentTensor2D
        elseif N == 3
            @assert typeof(momtens) <: MomentTensor3D
        else
            error("MomentTensorSources: Moment tensor neither 2D nor 3D.")
        end
        return new{N, T}(positions, tf, momtens, domfreq)
    end
end

# Default type constructor {Float64}
MomentTensorSources(positions, tf, momtens, domfreq) = begin
    if typeof(momtens) <: MomentTensor2D
        ndim = 2
    elseif typeof(momtens) <: MomentTensor3D
        ndim = 3
    end
    MomentTensorSources{ndim, Float64}(positions, tf, momtens, domfreq)
end

####################################################
