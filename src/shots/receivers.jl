@doc """
$(TYPEDEF)

Type representing a multi-receiver configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarReceivers{T} <: Receivers{T}
    "Receiver positions"
    positions::Matrix{T}
    "Array holding seismograms (as columns)"
    seismograms::Matrix{T}
    "Array holding observed seismograms (as columns)"
    observed::Matrix{T}
    "Inverse of the covariance matrix"
    invcov::AbstractMatrix{T}
    "Windows of data used for misfit calculations"
    windows::Vector{Pair{Int, Int}}

    @doc """
         ScalarReceivers(
           positions::Matrix{T},
           nt::Int;
           observed::Union{Matrix{T}, Nothing}=nothing,
           invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
           windows::Union{Vector{Pair{Int,Int}}, Nothing}=nothing
         ) where {T}$(TYPEDSIGNATURES)

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function ScalarReceivers(
        positions::Matrix{T},
        nt::Int;
        observed::Union{Matrix{T}, Nothing}=nothing,
        invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
        windows::Union{Vector{Pair{Int, Int}}, Nothing}=nothing
    ) where {T}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, size(positions, 1))
        if observed !== nothing
            @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        else
            observed = zeros(T, 0, 0)
        end
        if invcov !== nothing
            @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        else
            invcov = zeros(T, 0, 0)
        end
        if windows !== nothing
            for wnd in windows
                @assert 1 <= wnd.first <= wnd.second <= nt "Window $(wnd) is not consistent!"
            end
        else
            windows = []
        end
        return new{T}(positions, seismograms, observed, invcov, windows)
    end
end

############################################################################

@doc """
$(TYPEDEF)

Type representing a multi-receiver configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct VectorReceivers{T, N} <: Receivers{T}
    "Receiver positions"
    positions::Matrix{T}
    "Array holding seismograms (as columns)"
    seismograms::Array{T, 3}
    "Array holding observed seismograms (as columns)"
    observed::Array{T, 3}
    "Inverse of the covariance matrix"
    invcov::AbstractMatrix{T}
    "Windows of data used for misfit calculations"
    windows::Vector{Pair{Int, Int}}

    function VectorReceivers(
        positions::Matrix{T},
        nt::Int,
        ndim::Int=2;
        observed::Union{Array{T}, Nothing}=nothing,
        invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
        windows::Union{Vector{Pair{Int,Int}}, Nothing}=nothing
    ) where {T}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, ndim, size(positions, 1))  ## N+1 !!!  <<<<<<<<--------------<<<<
        if observed !== nothing
            @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        else
            observed = zeros(T, 0, 0, 0) # (nt,ndim,npos)
        end
        if invcov !== nothing
            @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        else
            invcov = zeros(T, 0, 0)
        end
        if windows !== nothing
            for wnd in windows
                @assert 1 <= wnd.first <= wnd.second <= nt "Window $(wnd) is not consistent!"
            end
        else
            windows = []
        end
        return new{T, ndim}(positions, seismograms, observed, invcov, windows)
    end
end
