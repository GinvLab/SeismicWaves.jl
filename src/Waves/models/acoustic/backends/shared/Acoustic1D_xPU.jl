@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it)
    isrc = floor(Int, possrcs[is,1])
    pnew[isrc] += dt2srctf[it,is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pnew, traces, posrecs, it)
    irec = floor(Int, posrecs[ir,1])
    traces[it,ir] = pnew[irec]

    return nothing
end

@parallel_indices (i) function update_p!(pold, pcur, pnew, fact, _dx2)
    d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])*_dx2
    pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2)

    return nothing
end

@views function forward_onestep!(
    pold, pcur, pnew, fact, dx,
    possrcs, dt2srctf, posrecs, traces, it
)
    nx = length(pcur)
    _dx2 = 1 / dx^2

    @parallel (2:nx-1) update_p!(pold, pcur, pnew, fact, _dx2)
    @parallel (1:size(possrcs,1)) inject_sources!(pnew, dt2srctf, possrcs, it)
    @parallel (1:size(posrecs,1)) record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end