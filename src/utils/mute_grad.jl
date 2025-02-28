

function mutearoundmultiplepoints!(inparr::Array{T,N},xyzpts::Matrix{T},grid::UniformFiniteDifferenceGrid{N,T},
                                   radiuspx::Integer) where {T,N}
    for i=1:size(xyzpts,1)
        mutearoundpoint!(inparr,xyzpts[i,:],grid,radiuspx)
    end
    return
end



function mutearoundpoint!(inparr::Array{T,N},xyzpt::Vector{T},grid::UniformFiniteDifferenceGrid{N,T},
                          radiuspx::Integer) where {T,N}

    @assert size(inparr) == grid.size
    rmax::T = radiuspx*maximum(NTuple(grid.spacing))

    # This is just for the future, in case an offset of the grid
    #   coordinates could be specified
    #gridinit = @SVector zeros(T,N)

    ijkmin = MVector{N,Int64}(undef)
    ijkmax = MVector{N,Int64}(undef)
    for d=1:N
        # check that the point is inside the grid
        # with offset
        # if gridinit[d] <= xyzpt[d] <= grid.extent[d]
        # no offset
        if zero(T) <= xyzpt[d] <= grid.extent[d]
            # with offset
            # xyzres = div(xyzpt-gridinit[d],grid.spacing[d])
            # no offset
            xyzres = div(xyzpt[d],grid.spacing[d])
            ijkpt = floor(Int64,xyzres) + 1 # .+1 julia indexing...
            # spatial limits for smoothing
            ijkmin[d] = ijkpt - radiuspx
            ijkmax[d] = ijkpt + radiuspx
        else
            error("mutearoundpoint!(): The point lies outside the grid on dimension $d at position $(xyzpt[d]).")
        end
    end

    inpcart = [ijkmin[d]:ijkmax[d] for d=1:N]
    xyzcur = zeros(N)
    caind = CartesianIndices(NTuple{N,UnitRange{Int64}}(inpcart))

    withinbounds::Bool = true
    for caind in caind
        idxs = caind.I

        for d=1:N
            withinbounds = 1 <= idxs[d] <= grid.size[d]
            if !withinbounds
                break
            end
            # with offset
            #xyzcur[d] = (idxs[d]-1)*grid.spacing[d] + gridinit[d]
            # no offset
            xyzcur[d] = (idxs[d]-1)*grid.spacing[d] 
        end

        if withinbounds
            # inverse of geometrical spreading
            r = sqrt(sum( (xyzpt .- xyzcur).^2 ))
            if r<=rmax 
                # normalized inverse of geometrical spreading
                att = r/rmax
                inparr[caind] *= att
            end
        end
    end    

    return nothing
end


