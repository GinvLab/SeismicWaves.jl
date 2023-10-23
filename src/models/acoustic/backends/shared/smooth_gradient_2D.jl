function smooth_gradient!(grad, possrcs, radius)
    nx, ny = size(grad)
    nsources = size(possrcs,1)
    for s in 1:nsources
        isrc = possrcs[s,1]
        jsrc = possrcs[s,2]
        imin = isrc - radius
        imax = isrc + radius
        jmin = jsrc - radius
        jmax = jsrc + radius
        for i in imin:imax
            for j in jmin:jmax
                if 1 <= i <= nx && 1 <= j <= ny
                    r = sqrt((i - isrc)^2 + (j - jsrc)^2)
                    if r < radius
                        grad[i, j] *= r / radius
                    end
                end
            end
        end
    end
end