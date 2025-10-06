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
    # "Array holding observed seismograms (as columns)"
    # observed::Matrix{T}
    # "Inverse of the covariance matrix"
    # invcov::AbstractMatrix{T}
    # "Windows of data used for misfit calculations"
    # windows::Vector{Pair{Int, Int}}

    @doc """
        $(TYPEDSIGNATURES)

    Create a single shot wave propagation receivers configuration from receiver positions.
    """
    function ScalarReceivers(
        positions::Matrix{T},
        nt::Int;
        # observed::Union{Matrix{T}, Nothing}=nothing,
        # invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
        # windows::Union{Vector{Pair{Int, Int}}, Nothing}=nothing
    ) where {T}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, size(positions, 1))
        # if observed !== nothing
        #     @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        # else
        #     observed = zeros(T, 0, 0)
        # end
        # if invcov !== nothing
        #     @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        # else
        #     invcov = zeros(T, 0, 0)
        # end
        # if windows !== nothing
        #     for wnd in windows
        #         @assert 1 <= wnd.first <= wnd.second <= nt "Window $(wnd) is not consistent!"
        #     end
        # else
        #     windows = []
        # end
        return new{T}(positions, seismograms) #, observed, invcov, windows)
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
    # "Array holding observed seismograms (as columns)"
    # observed::Array{T, 3}
    # "Inverse of the covariance matrix"
    # invcov::AbstractMatrix{T}
    # "Windows of data used for misfit calculations"
    # windows::Vector{Pair{Int, Int}}

    @doc """
        $(TYPEDSIGNATURES)

    Create a single shot wave propagation receivers configuration from receiver positions.
    """
    function VectorReceivers(
        positions::Matrix{T},
        nt::Int,
        ndim::Int=2;
        # observed::Union{Array{T}, Nothing}=nothing,
        # invcov::Union{AbstractMatrix{T}, Nothing}=nothing,
        # windows::Union{Vector{Pair{Int, Int}}, Nothing}=nothing
    ) where {T}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        seismograms = zeros(T, nt, ndim, size(positions, 1))  ## N+1 !!!  <<<<<<<<--------------<<<<
        # if observed !== nothing
        #     @assert size(seismograms) == size(observed) "Size of observed data is not (# timesteps, # receivers)!"
        # else
        #     observed = zeros(T, 0, 0, 0) # (nt,ndim,npos)
        # end
        # if invcov !== nothing
        #     @assert size(invcov) == (nt, nt) "Size of invcov is not (# timesteps, # timesteps)!"
        # else
        #     invcov = zeros(T, 0, 0)
        # end
        # if windows !== nothing
        #     for wnd in windows
        #         @assert 1 <= wnd.first <= wnd.second <= nt "Window $(wnd) is not consistent!"
        #     end
        # else
        #     windows = []
        # end
        return new{T, ndim}(positions, seismograms) #, observed, invcov, windows)
    end
end

struct VectorCrossCorrelationsReceivers{T, N} <: Receivers{T}
    "Receiver positions"
    positions::Matrix{T}
    "Array holding cross-correlations (as columns)"
    crosscorrelations::Array{T, 5}
    "Indices of the reference receivers"
    refs::Vector{Int}
    "Components to compute cross-correlations for"
    comps::Vector{Int}

    @doc """
        $(TYPEDSIGNATURES)

    Create a multi-receiver configuration for cross-correlation analysis from reference receivers to target receivers.
    """
    function VectorCrossCorrelationsReceivers(
        positions::Matrix{T},
        nt::Int,
        refs::Vector{Int},
        comps::Vector{Int}=[1, 2],
        ndim::Int=2
    ) where {T}
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        @assert all(r -> 1 <= r <= size(positions, 1), refs) "Reference receiver indices must be valid!"
        @assert all(c -> 1 <= c <= ndim, comps) "Components must be valid!"
        nrecs = size(positions, 1)
        nrefrecs = length(refs)
        crosscorrelations = zeros(T, nt*2 + 1, length(comps), ndim, nrefrecs, nrecs) # (time -T:T, cc1, cc2, idx ref rec, idx target rec)
        new{T, ndim}(positions, crosscorrelations, refs, comps)
    end
end