"""
$(TYPEDEF)

Type representing a multi-source configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarSources{T <: Real} <: Sources
    "Source positions"
    positions::Matrix{<:Real}
    "Source time function"
    tf::Matrix{T}
    "Dominant frequency"
    domfreq::T
    
    @doc """ 
        ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}
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
