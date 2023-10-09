macro d_dx_4th(a, i, j)
    return esc( :( ( -$a[$i+1, $j] + 27.0 * $a[$i, $j] - 27.0 * $a[$i-1, $j] + $a[$i-2, $j] ) ) )
end

macro d_dy_4th(a, i, j)
    return esc( :( ( -$a[$i, $j+1] + 27.0 * $a[$i, $j] - 27.0 * $a[$i, $j-1] + $a[$i, $j-2] ) ) )
end

@parallel_indices (is) function inject_sources!(pcur, srctf, possrcs, it)
    isrc = floor(Int, possrcs[is, 1])
    jsrc = floor(Int, possrcs[is, 2])
    pcur[isrc, jsrc] += srctf[it, is]

    return nothing
end

@parallel_indices (ir) function record_receivers!(pcur, traces, posrecs, it)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    traces[it, ir] = pcur[irec, jrec]

    return nothing
end

@parallel_indices (i,j) function update_p_CPML!(pcur, vx_cur, vy_cur, halo, fact_m0, nx, ny, _dx, _dy,
                                                ξ_x_l, ξ_x_r, a_x_l, a_x_r, b_x_l, b_x_r,
                                                ξ_y_l, ξ_y_r, a_y_l, a_y_r, b_y_l, b_y_r)
    # Compute velocity derivatives
    dv_dx = @d_dx_4th(vx_cur, i, j) * _dx
    dv_dy = @d_dy_4th(vy_cur, i, j) * _dy
    # Update CPML memory arrays if on the boundary
    if i <= halo + 1
        # left boundary
        ξ_x_l[i-1,j] = b_x_l[i-1] * ξ_x_l[i-1,j] + a_x_l[i-1] * dv_dx
        dv_dx += ξ_x_l[i-1,j]
    elseif i >= nx - halo
        # right boundary
        ii = i - (nx - halo) + 1
        ξ_x_r[ii,j] = b_x_r[ii] * ξ_x_r[ii,j] + a_x_r[ii] * dv_dx
        dv_dx += ξ_x_r[ii,j]
    end
    if j <= halo + 1
        # top boundary
        ξ_y_l[i,j-1] = b_y_l[j-1] * ξ_y_l[i,j-1] + a_y_l[j-1] * dv_dy
        dv_dy += ξ_y_l[i,j-1]
    elseif j >= ny - halo
        # bottom boundary
        jj = j - (ny - halo) + 1
        ξ_y_r[i,jj] = b_y_r[jj] * ξ_y_r[i,jj] + a_y_r[jj] * dv_dy
        dv_dy += ξ_y_r[i,jj]
    end
    # Update pressure
    pcur[i,j] -= fact_m0[i,j] * (dv_dx + dv_dy)

    return nothing
end

@parallel_indices (i,j) function update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
                                                 ψ_x_l, ψ_x_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    # Compute pressure derivative in x direction
    dp_dx = @d_dx_4th(pcur, i+1, j) * _dx
    # Update CPML memory arrays if on the boundary
    if i <= halo + 1
        # left boundary
        ψ_x_l[i,j] = b_x_hl[i] * ψ_x_l[i,j] + a_x_hl[i] * dp_dx
        dp_dx += ψ_x_l[i,j]
    elseif i >= nx - halo - 1
        # right boundary
        ii = i - (nx - halo - 1) + 1
        ψ_x_r[ii,j] = b_x_hr[ii] * ψ_x_r[ii,j] + a_x_hr[ii] * dp_dx
        dp_dx += ψ_x_r[ii,j]
    end
    # Update velocity
    vx_cur[i,j] -= fact_m1_x[i,j] * dp_dx

    return nothing
end

@parallel_indices (i,j) function update_vy_CPML!(pcur, vy_cur, halo, fact_m1_y, ny, _dy,
                                                 ψ_y_l, ψ_y_r, a_y_hl, a_y_hr, b_y_hl, b_y_hr)
    # Compute pressure derivative in y direction
    dp_dy = @d_dy_4th(pcur, i, j+1) * _dy
    # Update CPML memory arrays if on the boundary
    if j <= halo + 1
        # top boundary
        ψ_y_l[i,j] = b_y_hl[j] * ψ_y_l[i,j] + a_y_hl[j] * dp_dy
        dp_dy += ψ_y_l[i,j]
    elseif j >= ny - halo - 1
        # bottom boundary
        jj = j - (ny - halo - 1) + 1
        ψ_y_r[i,jj] = b_y_hr[jj] * ψ_y_r[i,jj] + a_y_hr[jj] * dp_dy
        dp_dy += ψ_y_r[i,jj]
    end
    # Update velocity
    vy_cur[i,j] -= fact_m1_y[i,j] * dp_dy

    return nothing
end

@parallel_indices (it, ir) function prescale_residuals_kernel!(residuals, posrecs, fact)
    irec = floor(Int, posrecs[ir, 1])
    jrec = floor(Int, posrecs[ir, 2])
    residuals[it, ir] *= fact[irec, jrec]

    return nothing
end

@views function prescale_residuals!(residuals, posrecs, fact)
    nrecs = size(posrecs, 1)
    nt = size(residuals, 1)
    @parallel (1:nt, 1:nrecs) prescale_residuals_kernel!(residuals, posrecs, fact)
end

@views function forward_onestep_CPML!(
    pcur, vx_cur, vy_cur, fact_m0, fact_m1_x, fact_m1_y, dx, dy, halo,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    a_y_l, a_y_r, a_y_hl, a_y_hr,
    b_x_l, b_x_r, b_x_hl, b_x_hr,
    b_y_l, b_y_r, b_y_hl, b_y_hr,
    possrcs, srctf, posrecs, traces, it;
    save_trace=true
)
    nx, ny = size(pcur)
    _dx = 1 / (dx * 24)
    _dy = 1 / (dy * 24)

    @parallel (3:(nx-2),3:(ny-2)) update_p_CPML!(pcur, vx_cur, vy_cur, halo, fact_m0, nx, ny, _dx, _dy,
                                                 ξ_x_l, ξ_x_r, a_x_l, a_x_r, b_x_l, b_x_r,
                                                 ξ_y_l, ξ_y_r, a_y_l, a_y_r, b_y_l, b_y_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)

    @parallel_async (2:(nx-2),1:ny) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
                                                    ψ_x_l, ψ_x_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    @parallel_async (1:nx,2:(ny-2)) update_vy_CPML!(pcur, vy_cur, halo, fact_m1_y, ny, _dy,
                                                    ψ_y_l, ψ_y_r, a_y_hl, a_y_hr, b_y_hl, b_y_hr)
    @synchronize

    if save_trace
        @parallel (1:size(posrecs, 1)) record_receivers!(pcur, traces, posrecs, it)
    end

end

@views function backward_onestep_CPML!(
    pcur, vx_cur, vy_cur, fact_m0, fact_m1_x, fact_m1_y, dx, dy, halo,
    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
    ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
    a_x_l, a_x_r, a_x_hl, a_x_hr,
    a_y_l, a_y_r, a_y_hl, a_y_hr,
    b_x_l, b_x_r, b_x_hl, b_x_hr,
    b_y_l, b_y_r, b_y_hl, b_y_hr,
    possrcs, srctf, it
)
    nx, ny = size(pcur)
    _dx = 1 / (dx * 24)
    _dy = 1 / (dy * 24)

    @parallel_async (2:(nx-2),1:ny) update_vx_CPML!(pcur, vx_cur, halo, fact_m1_x, nx, _dx,
                                                    ψ_x_l, ψ_x_r, a_x_hl, a_x_hr, b_x_hl, b_x_hr)
    @parallel_async (1:nx,2:(ny-2)) update_vy_CPML!(pcur, vy_cur, halo, fact_m1_y, ny, _dy,
                                                    ψ_y_l, ψ_y_r, a_y_hl, a_y_hr, b_y_hl, b_y_hr)
    @synchronize

    @parallel (3:(nx-2),3:(ny-2)) update_p_CPML!(pcur, vx_cur, vy_cur, halo, fact_m0, nx, ny, _dx, _dy,
                                                 ξ_x_l, ξ_x_r, a_x_l, a_x_r, b_x_l, b_x_r,
                                                 ξ_y_l, ξ_y_r, a_y_l, a_y_r, b_y_l, b_y_r)
    @parallel (1:size(possrcs, 1)) inject_sources!(pcur, srctf, possrcs, it)
    
end

@parallel function correlate_gradient_m1_kernel!(curgrad_m1_stag_x, curgrad_m1_stag_y, adjvcur_x, adjvcur_y, pold, _dx, _dy)
    @all(curgrad_m1_stag_x) = @all(curgrad_m1_stag_x) + (@all(adjvcur_x) * @d_xa(pold) * _dx)
    @all(curgrad_m1_stag_y) = @all(curgrad_m1_stag_y) + (@all(adjvcur_y) * @d_ya(pold) * _dy)
    
    return nothing
end