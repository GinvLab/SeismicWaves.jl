"""
Type representing a multi-receiver configuration for a wave propagation shot.
"""
struct Receivers{T<:Real}
    positions::Matrix{<:Real}
    seismograms::Matrix{T}
    observed::Matrix{T}

    @doc """
        Receivers[{T<:Real = Float64}](positions::Matrix{<:Real}, nt::Int, observed::Union{Matrix{T}, Nothing} = nothing)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function Receivers{T}(positions::Matrix{<:Real}, nt::Integer, observed::Union{Matrix{T}, Nothing} = nothing) where {T<:Real}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, size(positions, 1))
        if observed !== nothing
            @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        else
            observed = zero(seismograms)
        end
        new(positions, seismograms, observed)
    end
end

# Default type constructor
Receivers(positions, nt, observed=nothing) = Receivers{Float64}(positions, nt, observed)
