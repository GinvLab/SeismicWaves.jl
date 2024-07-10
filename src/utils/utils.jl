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

struct ArithmeticAverageInterpolation <: InterpolationMethod
    width::Int
end

interpolate(a::Array{T, N}, interp_method) where {T, N} = collect(interp(interp_method, a, i) for i in 1:N)

@views function interp(method::ArithmeticAverageInterpolation, a::Array{<:Real, N}, dim) where {N}
    return sum(
        a[CartesianIndices(Tuple(i == dim ? (j:size(a, i)+j-method.width) : (1:size(a, i)) for i in 1:N))] for j in 1:method.width
    ) ./ method.width
end

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

@views function find_nearest_grid_points(model::WaveSimulation{T}, positions::Matrix{T})::Matrix{Int} where {T}
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
  - `x0`: coordinate of source or receiver point
  - `b`: 'b' coefficient
  - `r`: cut-off radius
"""
function kaiser(x::Vector, x0::T, b::T, r::T) where {T}
    # Kaiser window function
    #  r is window half with
    # Rule of thumb for finite diff.:
    #   b=4.14  b=6.31
    #   r = 4.0*dx
    w = zeros(T, length(x))
    for i in 1:length(x)
        xcur = x[i] - x0
        if -r <= xcur <= r
            den = 1.0 / besseli(0, b)
            w[i] = den * besseli(0, b * (sqrt.(1 - (xcur / r)^2)))
        else
            w[i] = 0.0
        end
    end
    return w
end

#####################################################################

"""
Compute 1-D coefficients for windowed (Kaiser) sync interpolation of source or receiver position.

# Parameters

  - `xstart`: origin coordinate of the grid
  - `Δx`: grid spacing
  - `kind`: :monopole or :dipole, i.e., using sinc or derivative of sinc
  - `npts` (optional): half-number of grid points for the window
  - `beta` (optional): 'beta' parameter for the Kaiser windowing function
"""
function coeffsinc1D(xstart::Real, Δx::Real, xcenter::Real, kind::Symbol, nx::Int;
    npts::Int64=4, beta::Union{Nothing, Real}=nothing)
    ## Coefficients for sinc interpolation
    ##  in 1D
    ##  xstart is the x coordinate of first node in the regular grid    
    ##  
    ## Rule of thumb for npts==4 [Hicks 2002, Geophysics]:
    ##    beta=4.14 for monopoles,
    ##    beta=4.40 for dipoles 
    ###
    ### Julia:  sinc(x) =
    ###    \sin(\pi x) / (\pi x) if x \neq 0, and 1 if x = 0
    ###
    @assert xcenter >= xstart
    @assert xcenter <= (nx - 1) * Δx + xstart

    if beta === nothing
        if kind == :monopole
            beta = 4.14
        elseif kind == :dipole
            beta = 4.40
        end
    end

    radius = npts * Δx
    # Assuming x from grid starts at xstart
    xh = (xcenter - xstart) / Δx
    ix = floor(Int64, xh + 1)
    if mod((xcenter - xstart), Δx) == 0.0
        ixsta = ix - npts
        ixend = ix + npts
    else
        ixsta = ix - npts + 1
        ixend = ix + npts
    end
    x = [xstart + Δx * (i - 1) for i in ixsta:ixend]
    indexes = ixsta:ixend

    if kind == :monopole
        # interpolating sinc(x) = sin(pi*x)/(pi*x) see Julia definition
        intrpsinc = sinc.((x .- xcenter) ./ Δx) # factor 1/Δx ??

    elseif kind == :dipole
        # derivative of sinc
        # cosc(x) is the derivative of sinc(x) in Julia
        intrpsinc = cosc.((x .- xcenter) ./ Δx)

    else
        error("coeffsinc1d(): Wrong argument 'kind'.")
    end
    # apply Kaiser windowing
    kaix = kaiser(x, xcenter, beta, radius)
    itpfun = kaix .* intrpsinc
    # return also indices of window (as a range)
    return itpfun, indexes
end

#####################################################################

"""
Compute 2-D coefficients for windowed (Kaiser) sync interpolation of source or receiver position.

# Parameters

  - `xstart`, `zstart`: origin coordinates of the grid
  - `Δx`,`Δz` : grid spacing
  - `xcenter`, `zcenter`: coordinates of source or receiver point
  - `nx`,`nz`: grid size in x and y
  - `kind`: vector of symbols :monopole or :dipole, i.e., using sinc or derivative of sinc
  - `npts` (optional): half-number of grid points for the window
  - `beta` (optional): 'beta' parameter for the Kaiser windowing function
"""
function coeffsinc2D(xstart::Real, zstart::Real, Δx::Real, Δz::Real, xcenter::Real, zcenter::Real,
    nx::Int, nz::Int, kind::Vector{Symbol};
    npts::Int64=4, beta::Union{Nothing, Real}=nothing)

    ## Calculate the 2D array of coefficients
    xcoe, xidx = coeffsinc1D(xstart, Δx, xcenter, kind[1], nx; npts=npts, beta=beta)
    zcoe, zidx = coeffsinc1D(zstart, Δz, zcenter, kind[2], nz; npts=npts, beta=beta)

    function reflectcoeffsinc(coe, idx, nmax)
        #
        # "Reflect" coefficients past the edge, i.e., mirror and subtract them
        #  from the internal ones
        #
        if idx[1] < 1
            # We are before the edge
            nab = count(idx .< 1)
            # get the "reflected" coefficients
            reflcoe = coe[nab:-1:1]
            # Create a new set of indices excluding those above the surface
            idx = idx[nab+1:end]
            # Hicks 2002 Geophysics
            # Subtract coefficients past the edge
            coe[nab+1:2*nab] .-= reflcoe
            # Create a new set of coefficients excluding those above the surface
            coe = coe[nab+1:end]

        elseif idx[end] > nmax
            # We are past the edge
            nab = count(idx .> nmax)
            # get the "reflected" coefficients
            reflcoe = coe[end:-1:end-nab+1]
            # Create a new set of indices excluding those above the surface
            idx = idx[1:end-nab]
            # Hicks 2002 Geophysics
            # Subtract coefficients past the edge
            coe[end-2*nab+1:end-nab] .-= reflcoe
            # Create a new set of coefficients excluding those above the surface
            coe = coe[1:end-nab]
        end

        return coe, idx
    end

    ## Crop and reflect coefficients if they go beyond model edges [Hicks 2002, Geophysics]
    xcoe, xidx = reflectcoeffsinc(xcoe, xidx, nx)
    zcoe, zidx = reflectcoeffsinc(zcoe, zidx, nz)

    # tensor product of coeff. in x and z
    xzcoeff = zeros(typeof(xcenter), length(xcoe), length(zcoe))
    for j in eachindex(zcoe)
        for i in eachindex(xcoe)
            xzcoeff[i, j] = xcoe[i] * zcoe[j]
        end
    end

    return xidx, zidx, xzcoeff
end

#####################################################################

#####################################################################
