function smooth_gradient!(grad, possrcs, radius)
    nx = length(grad)
    nsources = size(possrcs,1)
    for s in 1:nsources
        isrc = possrcs[s,1]
        imin = isrc - radius
        imax = isrc + radius
        for i in imin:imax
            if 1 <= i <= nx
                r = abs(i - isrc)
                if r < radius
                    grad[i] *= r / radius
                end
            end
        end
    end
end