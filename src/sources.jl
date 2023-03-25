"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct Sources{T<:Real}
    positions::Matrix{<:Real}
    tf::Matrix{T}
    domfreq::Real

    @doc """
        Sources[{T<:Real = Float64}](
            positions::Matrix{<:Real},
            tf::Matrix{<:T},
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function Sources{T}(
        positions::Matrix{<:Real},
        tf::Matrix{T},
        domfreq::Real
    ) where {T<:Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        new(positions, tf, domfreq)
    end
end

# Default type constructor
Sources(positions, tf, domfreq) = Sources{Float64}(positions, tf, domfreq)
