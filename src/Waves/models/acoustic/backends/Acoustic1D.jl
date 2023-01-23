module Acoustic1D

# Dummy data module
module Data
    Array = Base.Array
end

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i = 1:nsrcs
        isrc = possrcs[i,1]
        pnew[isrc] += dt2srctf[it,i]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s = 1:nrecs
        irec = posrecs[s,1]
        traces[it,s] = pnew[irec]
    end
end

@views function update_p!(pold, pcur, pnew, fact, nx, _dx2)
    for i = 2:nx-1
        d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])*_dx2
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2)
    end
end

@views function forward_onestep!(
    pold, pcur, pnew, fact, dx,
    possrcs, dt2srctf, posrecs, traces, it
)
    nx = length(pcur)
    _dx2 = 1 / dx^2

    update_p!(pold, pcur, pnew, fact, nx, _dx2)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end

export forward_onestep!

end