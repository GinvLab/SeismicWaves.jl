function smooth_gradient!(grad, possrcs, radius)
    nx, ny, nz = size(grad)
    nsources = size(possrcs,1)
    for s in 1:nsources
        isrc = possrcs[s,1]
        jsrc = possrcs[s,2]
        ksrc = possrcs[s,3]
        imin = isrc - radius
        imax = isrc + radius
        jmin = jsrc - radius
        jmax = jsrc + radius
        kmin = ksrc - radius
        kmax = ksrc + radius
        for i in imin:imax
            for j in jmin:jmax
                for k in kmin:kmax
                    if 1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz
                        r = sqrt((i - isrc)^2 + (j - jsrc)^2 + (k - ksrc)^2)
                        if r <= radius
                            grad[i, j, k] *= r / radius
                        end
                    end
                end
            end
        end
    end
end