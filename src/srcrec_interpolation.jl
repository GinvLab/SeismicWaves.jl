

using SpecialFunctions

#####################################################################

"""

Kaiser windowing function.

# Parameters

- `x`: coordinate of points
- `x0`: coordinate of source or receiver point
- `b`: 'b' coefficient
- `r`: cut-off radius

"""
function kaiser(x::Vector, x0::Real, b::Real, r::Real)
    # Kaiser window function
    #  r is window half with
    # Rule of thumb for finite diff.:
    #   b=4.14  b=6.31
    #   r = 4.0*dx
    w=zeros(length(x))
    for i=1:length(x)
        xcur = x[i]-x0
        if -r<=xcur<=r 
            den = 1.0/besseli(0,b)
            w[i] = den*besseli(0, b*(sqrt.(1 -(xcur/r)^2)))
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
function coeffsinc1D(xstart::Real,Δx::Real,xcenter::Real, kind::Symbol ;
                     npts::Int64=4, beta::Union{Nothing,Real}=nothing)
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
    
    if beta==nothing
        if kind==:monopole
            beta = 4.14
        elseif kind==:dipole
            beta = 4.40
        end
    end
    
    radius = npts*Δx
    ## Assuming x from grid starts at xstart
    xh = (xcenter-xstart)/Δx
    ix = floor(Int64,xh+1)
    if mod((xcenter-xstart),Δx) == 0.0
        ixsta = ix-npts
        ixend = ix+npts
    else
        ixsta = ix-npts+1
        ixend = ix+npts
    end
    x = [xstart+Δx*(i-1) for i=ixsta:ixend]
    indexes = ixsta:ixend
    
    if kind==:monopole
        # interpolating sinc(x) = sin(pi*x)/(pi*x) see Julia definition
        intrpsinc = sinc.((x.-xcenter)./Δx) # factor 1/Δx ??
        
    elseif kind==:dipole
        # derivative of sinc
        # cosc(x) is the derivative of sinc(x) in Julia
        intrpsinc = cosc.((x.-xcenter)./Δx)

    else
        error("coeffsinc1d(): Wrong argument 'kind'.")
    end
    # apply Kaiser windowing
    kaix = kaiser(x,xcenter,beta,radius)
    itpfun = kaix.*intrpsinc
    # return also indices of window (as a range)
    return itpfun,indexes
end
# ## test:
# begin
#     xstart = 12.0
#     x0 = 15.75
#     Δx = 0.5
#     x = [(i-1)*Δx+xstart for i=1:15]
#     coe,idxs = coeffsinc1d(xstart,Δx,x0,:monopole)
#     scatterlines(idxs,coe)
#     @show idxs
#     Any[1:length(x) x]
# end

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
function coeffsinc2D(xstart::Real,zstart::Real,Δx::Real,Δz::Real,xcenter::Real,zcenter::Real,
                     nx::Integer,nz::Integer,kind::Vector{Symbol} ;
                     npts::Int64=4, beta::Union{Nothing,Real}=nothing)

    ## Calculate the 2D array of coefficients
    xcoe,xidx = coeffsinc1D(xstart,Δx,xcenter,kind[1],npts=npts,beta=beta)
    zcoe,zidx = coeffsinc1D(zstart,Δz,zcenter,kind[2],npts=npts,beta=beta)

    function reflectcoeffsinc(coe,idx,n)
        
        if idx[1] < 1
            # We are before the edge
            nab = count( idx.<1 )
            # get the "reflected" coefficients
            reflcoe = coe[nab:-1:1]
            # Create a new set of indices excluding those above the surface
            idx = idx[nab+1:end]
            # Hicks 2002 Geophysics
            # Subtract coefficients past the edge
            coe[nab+1:2*nab] .-= reflcoe
            # Create a new set of coefficients excluding those above the surface
            coe = coe[nab+1:end]
            
        elseif idx[end] > n
            # We are past the edge
            nab = count( idx.>n )
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
        
        return coe,idx
    end

    ## Crop coefficients if they go beyond model edges
    xcoe,xidx = reflectcoeffsinc(xcoe,xidx,nx)
    zcoe,zidx = reflectcoeffsinc(zcoe,zidx,nz)

    # tensor product of coeff. in x and z
    xzcoeff = zeros(typeof(xcenter),length(xcoe),length(zcoe))
    for j in eachindex(zcoe)
        for i in eachindex(xcoe)
           xzcoeff[i,j] = xcoe[i] * zcoe[j] 
        end
    end

    return xidx,zidx,xzcoeff
end

## test
# begin
#     include("../src/srcrec_interpolation.jl")
#     xstart = 12.0
#     x0 = 15.75
#     Δx = 0.5
#     zstart = 3.0
#     z0 = 6.12
#     Δz = 0.35
#     xidx,zidx,xzcoeff = coeffsinc2D(xstart,zstart,Δx,Δz,x0,z0,nx,nz,[:monopole,:monopole])
#     @show xidx
#     @show zidx
#     @show xzcoeff
#     fig,ax,hm=heatmap(xzcoeff); Colorbar(fig[1,2],hm); fig
# end
#####################################################################
