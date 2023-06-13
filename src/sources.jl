"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct ScalarSources{T <: Real} <: Sources
    positions::Matrix{<:Real}
    tf::Matrix{T}
    domfreq::T

    @doc """
        ScalarSources{T<:Real}( 
            positions::Matrix{<:Real},
            tf::Matrix{<:T},
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new(positions, tf, domfreq)
    end
end

# Default type constructor {Float64}
ScalarSources(positions, tf, domfreq) = ScalarSources{Float64}(positions, tf, domfreq)
