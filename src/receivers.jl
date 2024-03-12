@doc """
$(TYPEDEF)

Type representing a multi-receiver configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarReceivers{T <: Real} <: Receivers
    "Receiver positions"
    positions::Matrix{<:Real}
    "Array holding seismograms (as columns)"
    seismograms::Matrix{T}
    "Array holding observed seismograms (as columns)"
    observed::Matrix{T}
    "Inverse of the covariance matrix"
    invcov::AbstractMatrix{T}
    "Windows of data used for misfit calculations"
    windows::Vector{Pair{Int, Int}}

    @doc """
         ScalarReceivers{T}(
           positions::Matrix{<:Real},
           nt::Integer;
           observed::Union{Matrix{T}, Nothing}=nothing,
           invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
           windows::Union{Vector{Pair{Int,Int}}, Nothing}=nothing
         ) where {T <: Real}$(TYPEDSIGNATURES)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function ScalarReceivers{T}(
        positions::Matrix{<:Real},
        nt::Integer;
        observed::Union{Matrix{T}, Nothing}=nothing,
        invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
        windows::Union{Vector{Pair{Int, Int}}, Nothing}=nothing
    ) where {T <: Real}
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
        if windows !== nothing
            for wnd in windows
                @assert 1 <= wnd.first <= wnd.second <= nt "Window $(wnd) is not consistent!"
            end
        else
            windows = []
        end
        return new(positions, seismograms, observed, invcov, windows)
    end
end

# Default type constructor
@doc """
$(SIGNATURES)

Create a single shot wave propagation receivers configuration from receivers positions.
Default constructor for Float64.
"""
ScalarReceivers(positions, nt; observed=nothing, invcov=nothing, windows=nothing) =
    ScalarReceivers{Float64}(positions, nt; observed=observed, invcov=invcov, windows=windows)

@doc """
$(TYPEDEF)

Type representing a multi-receiver configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
# What about using the package ComputedFieldTypes.jl? @computed ...
struct VectorReceivers{N, T <: Real} <: Receivers
    "Receiver positions"
    positions::Matrix{T}
    "Array holding seismograms (as columns)"
    seismograms::Array{T, 3}
    "Array holding observed seismograms (as columns)"
    observed::Array{T, 3}
    "Inverse of the covariance matrix"
    invcov::AbstractMatrix{T}

    @doc raw"""
        Receivers[{T<:Real = Float64}](positions::Matrix{<:Real}, nt::Int, observed::Union{Matrix{T}, Nothing} = nothing)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function VectorReceivers{N, T}(
        positions::Matrix{T},
        nt::Integer,
        ndim::Integer=2;
        observed::Union{Array{T}, Nothing}=nothing,
        invcov::Union{AbstractMatrix{T}, Nothing}=nothing
    ) where {N, T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, ndim, size(positions, 1))  ## N+1 !!!  <<<<<<<<--------------<<<<
        if observed !== nothing
            @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        else
            observed = zeros(0, 0, 0) # (nt,ndim,npos)
        end
        if invcov !== nothing
            @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        else
            invcov = zeros(0, 0)
        end
        return new{ndim, T}(positions, seismograms, observed, invcov)
        #return new(positions, seismograms, observed, invcov)
    end
end

# Default type constructor
VectorReceivers(positions, nt, ndim; observed=nothing, invcov=nothing) = VectorReceivers{ndim, Float64}(positions, nt, ndim; observed=observed, invcov=invcov)
