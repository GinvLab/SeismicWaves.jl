@doc raw"""
Type representing a multi-receiver configuration for a wave propagation shot.
"""
struct Receivers{T<:Real}
    positions::Matrix{<:Real}
    seismograms::Matrix{T}
    observed::Matrix{T}
    invcov::AbstractMatrix{T}

    @doc raw"""
        Receivers[{T<:Real = Float64}](positions::Matrix{<:Real}, nt::Int, observed::Union{Matrix{T}, Nothing} = nothing)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function Receivers{T}(positions::Matrix{<:Real}, nt::Integer;
                          observed::Union{Matrix{T}, Nothing} = nothing, invcov::Union{AbstractMatrix{T}, Nothing} = nothing
    ) where {T<:Real}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, size(positions, 1))
        if observed !== nothing
            @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        else
            observed = zeros(0, 0)
        end
        if invcov !== nothing
            @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        else
            invcov = zeros(0, 0)
        end
        new(positions, seismograms, observed, invcov)
    end
end

# Default type constructor
Receivers(positions, nt; observed=nothing, invcov=nothing) = Receivers{Float64}(positions, nt; observed=observed, invcov=invcov)
