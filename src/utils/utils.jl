@doc """
$(TYPEDSIGNATURES)

Ricker source (second derivative of gaussian) source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
rickerstf(t::Real, t0::Real, f0::Real)::Real = (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-(pi * f0 * (t - t0))^2)

@doc """
$(TYPEDSIGNATURES) 

First derivative of gaussian source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
gaussderivstf(t::Real, t0::Real, f0::Real)::Real = (t - t0) * exp(-(pi * f0 * (t - t0))^2)

@doc """
$(TYPEDSIGNATURES) 

Gaussian source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
gaussstf(t::Real, t0::Real, f0::Real)::Real = -exp(-(pi * f0 * (t - t0))^2) / (2 * (pi * f0)^2)

"""
$(TYPEDSIGNATURES) 

Compute an optimal distribution of tasks (nsrc) for a given number of workers (threads).
Returns a vector of UnitRange object.
"""
function distribsrcs(nsrc::Int, nw::Int)
    ## calculate how to subdivide the srcs among the workers
    if nsrc >= nw
        dis = div(nsrc, nw)
        grpsizes = dis * ones(Int64, nw)
        resto = mod(nsrc, nw)
        if resto > 0
            ## add the reminder
            grpsizes[1:resto] .+= 1
        end
    else
        ## if more workers than sources use only necessary workers
        grpsizes = ones(Int64, nsrc)
    end
    ## now set the indices for groups of srcs
    grpsrc = UnitRange.(cumsum(grpsizes) .- grpsizes .+ 1, cumsum(grpsizes))
    return grpsrc
end

function find_nearest_grid_points(model::WaveSimulation{T}, positions::Matrix{T})::Matrix{Int} where {T}
    # source time functions
    nsrcs = size(positions, 1)                      # number of sources
    ncoos = size(positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    idx_positions = zeros(Int, size(positions))     # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [positions[s, i] / model.grid.spacing[i] + 1 for i in 1:ncoos]
        idx_positions[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    return idx_positions
end

####################################################################

"""
Kaiser windowing function.

# Parameters

  - `x`: coordinate of points
  - `β`: Kaiser shape coefficient
  - `r`: cut-off radius
"""
kaiser(x, r, β) = (-r <= x <= r ? besseli(0, β * sqrt(1 - (x / r)^2)) / besseli(0, β) : 0.0)

"""
Band limited delta function by combining Kaiser window and sinc function.

# Parameters

  - `x`: coordinate of points
  - `x0`: coordinate of delta function center
  - `r`: cut-off radius in number of grid points
  - `β`: Kaiser shape coefficient
  - `Δx`: grid spacing
"""
modd(x, x0, r, β, Δx) = begin
    res = kaiser(x - x0, r * Δx, β) * sinc((x - x0) / Δx)
    return res
end

"""
Compute coefficients for a 1D band limited delta function.

# Parameters

  - `x0`: coordinate of delta function center
  - `dx`: grid spacing
  - `nx`: number of grid points
  - `r`: cut-off radius in number of grid points
  - `β`: Kaiser shape coefficient
  - `xstart`: coordinate of the first grid point (from the left)
  - `mirror`: controls if the delta function is mirrored over the boundary or not
  - `xbl`: coordinate of left boundary
  - `xbr`: coordinate of right boundary
"""

function coeffsinc1D(x0::T, dx::T, nx::Int, r::Int, β::T, xstart::T, mirror::Bool, xbl::T, xbr::T) where {T}
    # Grid points
    xs = range(xstart; length=nx, step=dx)
    # Placeholders for indices and coefficients
    idxs, coeffs = Vector{Int}(), Vector{T}()
    # Find nearest grid point to the delta function center
    findnearest(x, xs) = argmin(abs.(x .- xs))
    i0 = findnearest(x0, xs)
    # Compute coefficients
    for idx in i0-r-1:i0+r+1
        xcurr = (idx-1) * dx + xstart
        coe = modd(xcurr, x0, r, β, dx)
        # Store coefficients that are not close to zero
        if !isapprox(coe, 0.0; atol=1e-15)
            push!(idxs, idx)
            push!(coeffs, coe)
        end
    end
    # Mirror or reflect coefficients over the boundary
    idxs_new, coeffs_new = Vector{Int}(), Vector{T}()
    for (idx, coeffs) in zip(idxs, coeffs)
        xcurr = (idx-1) * dx + xstart
        if xcurr < xbl
            idx_mirr = findnearest(xbl + (xbl - xcurr), xs)
            push!(idxs_new, idx_mirr)
            push!(coeffs_new, mirror ? -coeffs : coeffs)
        elseif xcurr > xbr
            idx_mirr = findnearest(xbr - (xcurr - xbr), xs)
            push!(idxs_new, idx_mirr)
            push!(coeffs_new, mirror ? -coeffs : coeffs)
        else
            push!(idxs_new, idx)
            push!(coeffs_new, coeffs)
        end
    end
    # Sum coefficients with same index
    idxs_dict = Dict{Int, T}()
    for (idx, coeff) in zip(idxs_new, coeffs_new)
        if haskey(idxs_dict, idx)
            idxs_dict[idx] += coeff
        else
            idxs_dict[idx] = coeff
        end
    end
    # Prepare final output
    idxs_new = collect(keys(idxs_dict))
    coeffs_new = collect(values(idxs_dict))

    return idxs_new, coeffs_new
end

"""
Computes coefficients and indices of the grid for arbitrarely placed sources or receivers using band limited delta functions coefficients.

# Parameters

  - `grid`: grid object
  - `positions`: matrix of source or receiver positions
  - `shifts`: shift of the source or receiver positions
  - `mirror`: controls if the delta function is mirrored over the boundary or not
  - `r`: cut-off radius in number of grid points (default = 4)
  - `β`: Kaiser shape coefficient (default = 6.31)
"""
function spread_positions(
    grid::SeismicWaves.UniformFiniteDifferenceGrid{N, T},
    positions::Matrix{T};
    shift::NTuple{N, T}=ntuple(_ -> zero(T), N),
    mirror::Bool=false,
    r::Int=4,
    β::T=T.(6.31),
    freesurfposition::Symbol
    ) where {N, T}
    
    # Position of the free surface
    if freesurfposition==:halfgridin
        position_freesurface = grid.spacing./2
    elseif freesurfposition==:ongridbound
        position_freesurface = zeros(T,N)
    else
        error("spread_positions(): Wrong keyword argument freesurfposition $freesurfposition")
    end
    # Get number of positions to spread
    npos = size(positions, 1)
    # Spread each position
    idxs = Vector{Matrix{Int}}(undef, npos)
    coeffs = Vector{Vector{T}}(undef, npos)
    for p in 1:npos
        # For each dimension
        idxs_dims = Vector{Vector{Int}}(undef, N)
        coeffs_dims = Vector{Vector{T}}(undef, N)
        for n in 1:N
            # Compute coefficients for the n-th dimension
            idxs_nth, coeffs_nth = coeffsinc1D(positions[p, n], grid.spacing[n], grid.size[n], r, β, shift[n], mirror, position_freesurface[n], grid.extent[n])
            
            idxs_dims[n] = idxs_nth
            coeffs_dims[n] = coeffs_nth
        end
        # Compute tensor product of indices and coefficients
        totlen = prod(length.(idxs_dims))
        idxs[p] = zeros(Int, totlen, N)
        coeffs[p] = zeros(T, totlen)
        for (ii, idx_comb) in enumerate(Iterators.product(idxs_dims...))
            idxs[p][ii, :] .= idx_comb
        end
        for (ii, coeff_comb) in enumerate(Iterators.product(coeffs_dims...))
            coeffs[p][ii] = prod(coeff_comb)
        end
    end
    return idxs, coeffs
end


