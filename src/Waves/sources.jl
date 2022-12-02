"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct Sources{T<:Real}
    position::Matrix{Int}
    tf::Matrix{Real}
    freqdomain::Real

    @doc """
        Sources[{T<:Real = Float64}](
            positions::Matrix{<:Int},
            tf::Matrix{<:T},
            freqdomain::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a frequency domain.
    """
    function Sources{T}(
        positions::Matrix{<:Int},
        tf::Matrix{<:T},
        freqdomain::T
    ) where {T<:Real}
        @assert size(positions, 1) > 0
        @assert size(positions, 1) == size(tf, 2)
        new(positions, tf, freqdomain)
    end

    # Default type constructor
    Sources(positions, tf, freqdomain) = Sources{Float64}(positions, tf, freqdomain)
end
