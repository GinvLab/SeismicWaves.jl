
module Acoustic2D_CD_CPML_Serial

# Dummy data module
module Data
Array = Base.Array
end

# the backend needs these
ones = Base.ones
zeros = Base.zeros

function update_ψ_x!(ψ_x_l, ψ_x_r, pcur, halo, _dx, nx, a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr)
    ny = size(ψ_x_l, 2)
    for j in 1:ny
        for i in 1:(halo+1)
            ii = i + nx - halo - 2  # shift for right boundary pressure indices
            # left boundary
            ψ_x_l[i, j] = b_K_x_hl[i] * ψ_x_l[i, j] + a_x_hl[i] * (pcur[i+1, j] - pcur[i, j]) * _dx
            # right boundary
            ψ_x_r[i, j] = b_K_x_hr[i] * ψ_x_r[i, j] + a_x_hr[i] * (pcur[ii+1, j] - pcur[ii, j]) * _dx
        end
    end
end

function update_ψ_y!(ψ_y_l, ψ_y_r, pcur, halo, _dy, ny, a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr)
    nx = size(ψ_y_l, 1)
    for j in 1:(halo+1)
        for i in 1:nx
            jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
            # top boundary
            ψ_y_l[i, j] = b_K_y_hl[j] * ψ_y_l[i, j] + a_y_hl[j] * (pcur[i, j+1] - pcur[i, j]) * _dy
            # bottom boundary
            ψ_y_r[i, j] = b_K_y_hr[j] * ψ_y_r[i, j] + a_y_hr[j] * (pcur[i, jj+1] - pcur[i, jj]) * _dy
        end
    end
end

function update_p_CPML!(
    pold, pcur, pnew, halo, fact,
    _dx, _dx2, _dy, _dy2, nx, ny,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r
)
    for j in 2:(ny-1)
        for i in 2:(nx-1)

            # pressure derivatives in space
            d2p_dx2 = (pcur[i+1, j] - 2.0 * pcur[i, j] + pcur[i-1, j]) * _dx2
            d2p_dy2 = (pcur[i, j+1] - 2.0 * pcur[i, j] + pcur[i, j-1]) * _dy2

            damp = 0.0
            # x boundaries
            if i <= halo + 1
                # left boundary
                dψ_x_dx = (ψ_x_l[i, j] - ψ_x_l[i-1, j]) * _dx
                ξ_x_l[i-1, j] = b_K_x_l[i-1] * ξ_x_l[i-1, j] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
                damp += fact[i, j] * (dψ_x_dx + ξ_x_l[i-1, j])
            elseif i >= nx - halo
                # right boundary
                ii = i - (nx - halo) + 2
                dψ_x_dx = (ψ_x_r[ii, j] - ψ_x_r[ii-1, j]) * _dx
                ξ_x_r[ii-1, j] = b_K_x_r[ii-1] * ξ_x_r[ii-1, j] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
                damp += fact[i, j] * (dψ_x_dx + ξ_x_r[ii-1, j])
            end
            # y boundaries
            if j <= halo + 1
                # top boundary
                dψ_y_dy = (ψ_y_l[i, j] - ψ_y_l[i, j-1]) * _dy
                ξ_y_l[i, j-1] = b_K_y_l[j-1] * ξ_y_l[i, j-1] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
                damp += fact[i, j] * (dψ_y_dy + ξ_y_l[i, j-1])
            elseif j >= ny - halo
                # bottom boundary
                jj = j - (ny - halo) + 2
                dψ_y_dy = (ψ_y_r[i, jj] - ψ_y_r[i, jj-1]) * _dy
                ξ_y_r[i, jj-1] = b_K_y_r[jj-1] * ξ_y_r[i, jj-1] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
                damp += fact[i, j] * (dψ_y_dy + ξ_y_r[i, jj-1])
            end

            # update pressure
            pnew[i, j] = 2.0 * pcur[i, j] - pold[i, j] + fact[i, j] * (d2p_dx2 + d2p_dy2) + damp
        end
    end
end

inject_sources!(pnew, dt2srctf, possrcs, it) =
    for is in axes(possrcs, 1)
        isrc = floor(Int, possrcs[is, 1])
        jsrc = floor(Int, possrcs[is, 2])
        pnew[isrc, jsrc] += dt2srctf[it, is]
    end

record_receivers!(pnew, traces, posrecs, it) =
    for ir in axes(posrecs, 1)
        irec = floor(Int, posrecs[ir, 1])
        jrec = floor(Int, posrecs[ir, 2])
        traces[it, ir] = pnew[irec, jrec]
    end

prescale_residuals!(residuals, possrcs, fact) =
    for is in axes(possrcs, 1)
        isrc = floor(Int, possrcs[is, 1])
        jsrc = floor(Int, possrcs[is, 2])
        for it in axes(residuals, 1) # nt
            residuals[it, is] *= fact[isrc, jsrc]
        end
    end

function forward_onestep_CPML!(
    pold, pcur, pnew, fact, dx, dy, halo,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    a_y_l, a_y_r, a_y_hl, a_y_hr,
    b_K_x_l, b_K_x_r, b_K_x_hl, b_K_x_hr,
    b_K_y_l, b_K_y_r, b_K_y_hl, b_K_y_hr,
    possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    nx, ny = size(pcur)
    _dx = 1 / dx
    _dx2 = 1 / dx^2
    _dy = 1 / dy
    _dy2 = 1 / dy^2

    # update ψ arrays
    update_ψ_x!(ψ_x_l, ψ_x_r, pcur, halo, _dx, nx, a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr)

    update_ψ_y!(ψ_y_l, ψ_y_r, pcur, halo, _dy, ny, a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr)

    # update pressure and ξ arrays
    update_p_CPML!(pold, pcur, pnew, halo, fact,
        _dx, _dx2, _dy, _dy2, nx, ny,
        ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
        ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
        a_x_l, a_x_r, b_K_x_l, b_K_x_r,
        a_y_l, a_y_r, b_K_y_l, b_K_y_r)

    # inject sources
    inject_sources!(pnew, dt2srctf, possrcs, it)
    # record receivers
    if save_trace
        record_receivers!(pnew, traces, posrecs, it)
    end

    return pcur, pnew, pold
end

#########################################

function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    nx, ny = size(curgrad)
    for j in 1:ny
        for i in 1:nx
            curgrad[i, j] = curgrad[i, j] + (adjcur[i, j] * (pcur[i, j] - 2.0 * pold[i, j] + pveryold[i, j]) * _dt2)
        end
    end
end

#########################################
end  # end module
#########################################
