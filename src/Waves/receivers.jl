"""
Type representing a multi-receiver configuration for a wave propagation shot.
"""
struct Receivers{T<:Real}
    positions::Matrix{Int}
    seismograms::Matrix{T}

    @doc """
        Receivers[{T<:Real = Float64}](positions::Matrix{<:Int}, nt::Int)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function Receivers{T}(positions::Matrix{<:Int}, nt::Int) where {T<:Real}
        @assert size(positions, 1) > 0
        seismograms = zeros(T, nt, size(positions, 1))
        new(positions, seismograms)
    end

    # Default type constructor
    Receivers(positions, nt) = Receivers{Float64}(positions, nt)
end
