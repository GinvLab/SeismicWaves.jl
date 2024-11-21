@parallel_indices (i, j) function update_4thord_vx!(
    vx, σxx, σxz, dt, _dx, _dz, ρ_ihalf,
    halo, ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop
)

    ####################################
    # ∂vx/∂t = (∂σxx/∂x + ∂σxz/∂z) / ρ #
    ####################################

    # Compute partial derivatives
    ∂σxx∂x     = @∂̃x(σxx, a_x_half, b_x_half, ψ_∂σxx∂x,
                     order=4, I=(i,j), _Δ=_dx,
                     halo=halo, halfgrid=true)
    if freetop  # if free surface, mirror the top boundary
        ∂σxz∂z = @∂̃y(σxz, a_z, b_z, ψ_∂σxz∂z,
                     order=4, I=(i,j-1), _Δ=_dz, mirror=(true, false),
                     halo=halo, halfgrid=false)
    else
        ∂σxz∂z = @∂̃y(σxz, a_z, b_z, ψ_∂σxz∂z,
                     order=4, I=(i,j-1), _Δ=_dz,
                     halo=halo, halfgrid=false)
    end

    # Update velocity
    vx[i, j] += dt * (∂σxx∂x + ∂σxz∂z) / ρ_ihalf[i, j]

    return nothing
end

@parallel_indices (i, j) function update_4thord_adjvx!(
    adjvx, adjσxx, adjσzz, adjσxz, dt, _dx, _dz, ρ_ihalf, λ_ihalf, μ_ihalf,
    halo, adjψ_∂σxx∂x, adjψ_∂σzz∂x, adjψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z
)

    ###############################################################################
    # ∂adjvx/∂t = ((λ + 2μ)/ρ * ∂adjσxx/∂x + λ/ρ * ∂adjσzz/∂x + μ/ρ * ∂adjσxz/∂z) #
    ###############################################################################

    # Compute partial derivatives
    ∂adjσxx∂x = @∂̃x(adjσxx, a_x_half, b_x_half, adjψ_∂σxx∂x,
                    order=4, I=(i,j), _Δ=_dx,
                    halo=halo, halfgrid=true)
    ∂adjσzz∂x = @∂̃x(adjσzz, a_x_half, b_x_half, adjψ_∂σzz∂x,
                    order=4, I=(i,j), _Δ=_dx,
                    halo=halo, halfgrid=true)
    ∂adjσxz∂z = @∂̃y(adjσxz, a_z, b_z, adjψ_∂σxz∂z,
                    order=4, I=(i,j-1), _Δ=_dz,
                    halo=halo, halfgrid=false)

    # Update adjoint velocity
    adjvx[i, j] += dt * (((λ_ihalf[i, j] + 2*μ_ihalf[i, j]) / ρ_ihalf[i, j]) * ∂adjσxx∂x +
                         ( λ_ihalf[i, j]                    / ρ_ihalf[i, j]) * ∂adjσzz∂x +
                         ( μ_ihalf[i, j]                    / ρ_ihalf[i, j]) * ∂adjσxz∂z)

    return nothing
end

@parallel_indices (i, j) function update_4thord_vz!(
    vz, σxz, σzz, dt, _dx, _dz, ρ_jhalf,
    halo, ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop
)

    ####################################
    # ∂vz/∂t = (∂σzz/∂z + ∂σxz/∂x) / ρ #
    ####################################

    # Compute partial derivatives
    ∂σxz∂x     = @∂̃x(σxz, a_x, b_x, ψ_∂σxz∂x,
                     order=4, I=(i-1,j), _Δ=_dx,
                     halo=halo, halfgrid=false)
    if freetop # if free surface, mirror the top boundary
        ∂σzz∂z = @∂̃y(σzz, a_z_half, b_z_half, ψ_∂σzz∂z,
                     order=4, I=(i,j), _Δ=_dz, mirror=(true, false),
                     halo=halo, halfgrid=true)
    else
        ∂σzz∂z = @∂̃y(σzz, a_z_half, b_z_half, ψ_∂σzz∂z,
                     order=4, I=(i,j), _Δ=_dz,
                     halo=halo, halfgrid=true)
    end

    # Update velocity
    vz[i, j] += dt * (∂σzz∂z + ∂σxz∂x) / ρ_jhalf[i, j]

    return nothing
end

@parallel_indices (i, j) function update_4thord_adjvz!(
    adjvz, adjσxx, adjσzz, adjσxz, dt, _dx, _dz, ρ_jhalf, λ_jhalf, μ_jhalf,
    halo, adjψ_∂σxx∂z, adjψ_∂σzz∂z, adjψ_∂σxz∂x, b_x, b_z_half, a_x, a_z_half
)

    ###############################################################################
    # ∂adjvz/∂t = ((λ + 2μ)/ρ * ∂adjσzz/∂z + λ/ρ * ∂adjσxx/∂z + μ/ρ * ∂adjσxz/∂x) #
    ###############################################################################

    # Compute partial derivatives
    ∂adjσxx∂z = @∂̃y(adjσxx, a_z_half, b_z_half, adjψ_∂σxx∂z,
                    order=4, I=(i,j), _Δ=_dz,
                    halo=halo, halfgrid=true)
    ∂adjσzz∂z = @∂̃y(adjσzz, a_z_half, b_z_half, adjψ_∂σzz∂z,
                    order=4, I=(i,j), _Δ=_dz,
                    halo=halo, halfgrid=true)
    ∂adjσxz∂x = @∂̃x(adjσxz, a_x, b_x, adjψ_∂σxz∂x,
                    order=4, I=(i-1,j), _Δ=_dx,
                    halo=halo, halfgrid=false)

    # Update adjoint velocity
    adjvz[i, j] += dt * (((λ_jhalf[i, j] + 2*μ_jhalf[i, j]) / ρ_jhalf[i, j]) * ∂adjσzz∂z +
                         ( λ_jhalf[i, j]                    / ρ_jhalf[i, j]) * ∂adjσxx∂z +
                         ( μ_jhalf[i, j]                    / ρ_jhalf[i, j]) * ∂adjσxz∂x)

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxxσzz!(
    σxx, σzz, vx, vz, dt, _dx, _dz, λ, μ,
    halo, ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop
)

    ############################################################
    # ∂vz/∂z  = -λ/(λ + 2μ) * ∂vx/∂x if z = 0 and free surface #
    # σzz     = 0                    if z = 0 and free surface #
    # ∂σxx/∂t = ((λ + 2μ) * ∂vx/∂x + μ * ∂vz/∂z)               #
    # ∂σzz/∂t = (λ * ∂vx/∂x + (λ + 2μ) * ∂vz/∂z)               #
    ############################################################

    # Compute partial derivatives
    ∂vx∂x     = @∂̃x(vx, a_x, b_x, ψ_∂vx∂x,
                    order=4, I=(i-1,j), _Δ=_dx,
                    halo=halo, halfgrid=false)
    if freetop && j == 1  # on the free surface, using BDC to calculate ∂vz∂z
        ∂vz∂z = -(λ[i, j] / (λ[i, j] + 2*μ[i, j])) * ∂vx∂x
    else
        ∂vz∂z = @∂̃y(vz, a_z, b_z, ψ_∂vz∂z,
                    order=4, I=(i,j-1), _Δ=_dz,
                    halo=halo, halfgrid=false)
    end

    # Update stresses
    σxx[i, j]     += dt * ((λ[i, j] + 2*μ[i, j]) * ∂vx∂x +  λ[i, j]              * ∂vz∂z)
    if freetop && j == 1 # on the free surface, σzz = 0
        σzz[i, j] = 0.0
    else
        σzz[i, j] += dt * ( λ[i, j]              * ∂vx∂x + (λ[i, j] + 2*μ[i, j]) * ∂vz∂z)
    end

    return nothing
end

@parallel_indices (i, j) function update_4thord_adjσxxσzz!(
    adjσxx, adjσzz, adjvx, adjvz, dt, _dx, _dz,
    halo, adjψ_∂vx∂x, adjψ_∂vz∂z, b_x, b_z, a_x, a_z
)

    ##########################
    # ∂adjσxx/∂t = ∂adjvx/∂x #
    # ∂adjσzz/∂t = ∂adjvz/∂z #
    ##########################    

    # Compute partial derivatives
    ∂adjvx∂x = @∂̃x(adjvx, a_x, b_x, adjψ_∂vx∂x,
                   order=4, I=(i-1,j), _Δ=_dx,
                   halo=halo, halfgrid=false)
    ∂adjvz∂z = @∂̃y(adjvz, a_z, b_z, adjψ_∂vz∂z,
                   order=4, I=(i,j-1), _Δ=_dz,
                   halo=halo, halfgrid=false)

    # Update adjoint stresses
    adjσxx[i, j] += dt * ∂adjvx∂x
    adjσzz[i, j] += dt * ∂adjvz∂z

    return nothing
end

@parallel_indices (i, j) function update_4thord_σxz!(
    σxz, vx, vz, dt, _dx, _dz, μ_ihalf_jhalf,
    halo, ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half
)

    ####################################
    # ∂σxz/∂t = μ * (∂vx/∂z + ∂vz/∂x) #
    ####################################

    # Compute partial derivatives
    ∂vx∂z = @∂̃y(vx, a_z_half, b_z_half, ψ_∂vx∂z,
                order=4, I=(i,j), _Δ=_dz,
                halo=halo, halfgrid=true)
    ∂vz∂x = @∂̃x(vz, a_x_half, b_x_half, ψ_∂vz∂x,
                order=4, I=(i,j), _Δ=_dx,
                halo=halo, halfgrid=true)

    # Update stress
    σxz[i, j] += dt * μ_ihalf_jhalf[i, j] * (∂vx∂z + ∂vz∂x)

    return nothing
end

@parallel_indices (i, j) function update_4thord_adjσxz!(
    adjσxz, adjvx, adjvz, dt, _dx, _dz,
    halo, adjψ_∂vx∂z, adjψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half
)

    ######################################
    # ∂adjσxz/∂t = ∂adjvx/∂z + ∂adjvz/∂x #
    ######################################

    # Compute partial derivatives
    ∂adjvx∂z = @∂̃y(adjvx, a_z_half, b_z_half, adjψ_∂vx∂z,
                   order=4, I=(i,j), _Δ=_dz,
                   halo=halo, halfgrid=true)
    ∂adjvz∂x = @∂̃x(adjvz, a_x_half, b_x_half, adjψ_∂vz∂x,
                   order=4, I=(i,j), _Δ=_dx,
                   halo=halo, halfgrid=true)

    # Update adjoint stress
    adjσxz[i, j] += dt * (∂adjvx∂z + ∂adjvz∂x)

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx, Mzz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxx[isrc, jsrc] += Mxx[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt
    σzz[isrc, jsrc] += Mzz[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxz!(σxz, Mxz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxz[isrc, jsrc] += Mxz[s] * srccoeval_bk[p] * srctf_bk[it, s] * dt

    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_ihalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    vx[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 1, s] / ρ_ihalf[isrc, jsrc] * dt
    return nothing
end

@parallel_indices (p) function inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_jhalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    vz[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 2, s] / ρ_jhalf[isrc, jsrc] * dt
    return nothing
end

@parallel_indices (p) function record_receivers2D_vx!(vx, traces_vx_bk_buf, reccoeij_vx, reccoeval_vx)
    irec, jrec = reccoeij_vx[p, 1], reccoeij_vx[p, 2]
    traces_vx_bk_buf[p] = reccoeval_vx[p] * vx[irec, jrec]
    return nothing
end

@parallel_indices (p) function record_receivers2D_vz!(vz, traces_vz_bk_buf, reccoeij_vz, reccoeval_vz)
    irec, jrec = reccoeij_vz[p, 1], reccoeij_vz[p, 2]
    traces_vz_bk_buf[p] = reccoeval_vz[p] * vz[irec, jrec]
    return nothing
end

function forward_onestep_CPML!(
    model,
    srccoeij_xx,
    srccoeval_xx,
    srccoeij_xz,
    srccoeval_xz,
    reccoeij_vx,
    reccoeval_vx,
    reccoeij_vz,
    reccoeval_vz,
    srctf_bk,
    reduced_buf,
    traces_vx_bk_buf,
    traces_vz_bk_buf,
    traces_bk,
    it::Int,
    Mxx_bk,
    Mzz_bk,
    Mxz_bk;
    save_trace::Bool=true
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
    cpmlcoeffs = model.cpmlcoeffs
    dx = model.grid.spacing[1]
    dz = model.grid.spacing[2]
    dt = model.dt
    nx, nz = model.grid.size[1:2]
    halo = model.cpmlparams.halo
    grid = model.grid

    vx, vz = grid.fields["v"].value
    σxx, σzz, σxz = grid.fields["σ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["ψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["ψ_∂σ∂z"].value
    ψ_∂vx∂x, ψ_∂vz∂x = grid.fields["ψ_∂v∂x"].value
    ψ_∂vx∂z, ψ_∂vz∂z = grid.fields["ψ_∂v∂z"].value

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    λ = grid.fields["λ"].value
    μ = grid.fields["μ"].value
    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = grid.fields["μ_ihalf_jhalf"].value

    # Precomputing divisions
    _dx = 1.0 / dx
    _dz = 1.0 / dz

    # update velocity vx
    @parallel (1:nx-1, 1:nz) update_4thord_vx!(
        vx, σxx, σxz, dt, _dx, _dz, ρ_ihalf,
        halo, ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop
    )
    # update velocity vz
    @parallel (1:nx, 1:nz-1) update_4thord_vz!(
        vz, σxz, σzz, dt, _dx, _dz, ρ_jhalf,
        halo, ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop
    )

    # record receivers
    if save_trace
        nrecs = size(reccoeij_vx, 1)
        for r in 1:nrecs
            nrecspts_vx = size(reccoeij_vx[r], 1)
            nrecspts_vz = size(reccoeij_vz[r], 1)
            @parallel (1:nrecspts_vx) record_receivers2D_vx!(vx, traces_vx_bk_buf[r], reccoeij_vx[r], reccoeval_vx[r])
            @parallel (1:nrecspts_vz) record_receivers2D_vz!(vz, traces_vz_bk_buf[r], reccoeij_vz[r], reccoeval_vz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_vx_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_vz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf[1])
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf[2])
        reduced_buf[1] .= 0.0
        reduced_buf[2] .= 0.0
    end

    # update stresses σxx and σzz
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_4thord_σxxσzz!(
        σxx, σzz, vx, vz, dt, _dx, _dz, λ, μ,
        halo, ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop
    )
    # update stress σxz
    @parallel (1:nx-1, 1:nz-1) update_4thord_σxz!(
        σxz, vx, vz, dt, _dx, _dz, μ_ihalf_jhalf,
        halo, ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half
    )

    # inject sources (moment tensor type of internal force)
    nsrcs = size(srccoeij_xx, 1)
    for s in 1:nsrcs
        nsrcpts_xx = size(srccoeij_xx[s], 1)
        nsrcpts_xz = size(srccoeij_xz[s], 1)
        @parallel (1:nsrcpts_xx) inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx_bk, Mzz_bk, srctf_bk, srccoeij_xx[s], srccoeval_xx[s], it, s, dt)
        @parallel (1:nsrcpts_xz) inject_momten_sources2D_σxz!(σxz, Mxz_bk, srctf_bk, srccoeij_xz[s], srccoeval_xz[s], it, s, dt)
    end

    return
end

function forward_onestep_CPML!(
    model,
    srccoeij_vx,
    srccoeval_vx,
    srccoeij_vz,
    srccoeval_vz,
    reccoeij_vx,
    reccoeval_vx,
    reccoeij_vz,
    reccoeval_vz,
    srctf_bk,
    reduced_buf,
    traces_vx_bk_buf,
    traces_vz_bk_buf,
    traces_bk,
    it::Int;
    save_trace::Bool=true
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
    cpmlcoeffs = model.cpmlcoeffs
    dx = model.grid.spacing[1]
    dz = model.grid.spacing[2]
    dt = model.dt
    nx, nz = model.grid.size[1:2]
    halo = model.cpmlparams.halo
    grid = model.grid

    vx, vz = grid.fields["v"].value
    σxx, σzz, σxz = grid.fields["σ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["ψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["ψ_∂σ∂z"].value
    ψ_∂vx∂x, ψ_∂vz∂x = grid.fields["ψ_∂v∂x"].value
    ψ_∂vx∂z, ψ_∂vz∂z = grid.fields["ψ_∂v∂z"].value

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    λ = grid.fields["λ"].value
    μ = grid.fields["μ"].value
    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = grid.fields["μ_ihalf_jhalf"].value

    # Precomputing divisions
    _dx = 1.0 / dx
    _dz = 1.0 / dz

    # update velocity vx
    @parallel (1:nx-1, 1:nz) update_4thord_vx!(
        vx, σxx, σxz, dt, _dx, _dz, ρ_ihalf,
        halo, ψ_∂σxx∂x, ψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z, freetop
    )
    # update velocity vz
    @parallel (1:nx, 1:nz-1) update_4thord_vz!(
        vz, σxz, σzz, dt, _dx, _dz, ρ_jhalf,
        halo, ψ_∂σxz∂x, ψ_∂σzz∂z, b_x, b_z_half, a_x, a_z_half, freetop
    )

    # inject sources
    nsrcs = size(srccoeij_vx, 1)
    for s in 1:nsrcs
        nsrcpts_vx = size(srccoeij_vx[s], 1)
        nsrcpts_vz = size(srccoeij_vz[s], 1)
        @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(vx, srctf_bk, srccoeij_vx[s], srccoeval_vx[s], ρ_ihalf, it, s, dt)
        @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(vz, srctf_bk, srccoeij_vz[s], srccoeval_vz[s], ρ_jhalf, it, s, dt)
    end

    if save_trace
        nrecs = size(reccoeij_vx, 1)
        for r in 1:nrecs
            nrecspts_vx = size(reccoeij_vx[r], 1)
            nrecspts_vz = size(reccoeij_vz[r], 1)
            @parallel (1:nrecspts_vx) record_receivers2D_vx!(vx, traces_vx_bk_buf[r], reccoeij_vx[r], reccoeval_vx[r])
            @parallel (1:nrecspts_vz) record_receivers2D_vz!(vz, traces_vz_bk_buf[r], reccoeij_vz[r], reccoeval_vz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_vx_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_vz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf[1])
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf[2])
        reduced_buf[1] .= 0.0
        reduced_buf[2] .= 0.0
    end

    # update stresses σxx and σzz
    idxzσxx = freetop ? (1:nz-2) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_4thord_σxxσzz!(
        σxx, σzz, vx, vz, dt, _dx, _dz, λ, μ,
        halo, ψ_∂vx∂x, ψ_∂vz∂z, b_x, b_z, a_x, a_z, freetop
    )
    # update stress σxz
    @parallel (1:nx-1, 1:nz-1) update_4thord_σxz!(
        σxz, vx, vz, dt, _dx, _dz, μ_ihalf_jhalf,
        halo, ψ_∂vx∂z, ψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half
    )

    return
end

function adjoint_onestep_CPML!(
    model,
    srccoeij_vx,
    srccoeval_vx,
    srccoeij_vz,
    srccoeval_vz,
    residuals_bk,
    it
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
    cpmlcoeffs = model.cpmlcoeffs
    dx = model.grid.spacing[1]
    dz = model.grid.spacing[2]
    dt = model.dt
    nx, nz = model.grid.size[1:2]
    halo = model.cpmlparams.halo
    grid = model.grid

    adjvx, adjvz = grid.fields["adjv"].value
    adjσxx, adjσzz, adjσxz = grid.fields["adjσ"].value

    adjψ_∂σxx∂x, adjψ_∂σzz∂x, adjψ_∂σxz∂x = grid.fields["adjψ_∂σ∂x"].value
    adjψ_∂σxx∂z, adjψ_∂σzz∂z, adjψ_∂σxz∂z = grid.fields["adjψ_∂σ∂z"].value
    adjψ_∂vx∂x, adjψ_∂vz∂x = grid.fields["adjψ_∂v∂x"].value
    adjψ_∂vx∂z, adjψ_∂vz∂z = grid.fields["adjψ_∂v∂z"].value

    a_x = cpmlcoeffs[1].a
    a_x_half = cpmlcoeffs[1].a_h
    b_x = cpmlcoeffs[1].b
    b_x_half = cpmlcoeffs[1].b_h

    a_z = cpmlcoeffs[2].a
    a_z_half = cpmlcoeffs[2].a_h
    b_z = cpmlcoeffs[2].b
    b_z_half = cpmlcoeffs[2].b_h

    ρ_ihalf = grid.fields["ρ_ihalf"].value
    ρ_jhalf = grid.fields["ρ_jhalf"].value
    λ_ihalf = grid.fields["λ_ihalf"].value
    λ_jhalf = grid.fields["λ_jhalf"].value
    μ_ihalf = grid.fields["μ_ihalf"].value
    μ_jhalf = grid.fields["μ_jhalf"].value

    # Precomputing divisions
    _dx = 1.0 / dx
    _dz = 1.0 / dz
    
    # update adjoint stresses σxx and σzz 
    @parallel (1:nx-1, 1:nz-1) update_4thord_adjσxxσzz!(
        adjσxx, adjσzz, adjvx, adjvz, dt, _dx, _dz,
        halo, adjψ_∂vx∂x, adjψ_∂vz∂z, b_x, b_z, a_x, a_z
    )
    # update adjoint stress σxz
    @parallel (1:nx-1, 1:nz-1) update_4thord_adjσxz!(
        adjσxz, adjvx, adjvz, dt, _dx, _dz,
        halo, adjψ_∂vx∂z, adjψ_∂vz∂x, b_x_half, b_z_half, a_x_half, a_z_half
    )

    # update adjoint velocity vx
    @parallel (1:nx-1, 1:nz) update_4thord_adjvx!(
        adjvx, adjσxx, adjσzz, adjσxz, dt, _dx, _dz, ρ_ihalf, λ_ihalf, μ_ihalf,
        halo, adjψ_∂σxx∂x, adjψ_∂σzz∂x, adjψ_∂σxz∂z, b_x_half, b_z, a_x_half, a_z
    )
    # update adjoint velocity vz
    @parallel (1:nx, 1:nz-1) update_4thord_adjvz!(
        adjvz, adjσxx, adjσzz, adjσxz, dt, _dx, _dz, ρ_jhalf, λ_jhalf, μ_jhalf,
        halo, adjψ_∂σxx∂z, adjψ_∂σzz∂z, adjψ_∂σxz∂x, b_x, b_z_half, a_x, a_z_half
    )

    # inject sources (residuals as velocities)
    nsrcs = size(srccoeij_vx, 1)
    for s in 1:nsrcs
        nsrcpts_vx = size(srccoeij_vx[s], 1)
        nsrcpts_vz = size(srccoeij_vz[s], 1)
        @parallel (1:nsrcpts_vx) inject_external_sources2D_vx!(adjvx, residuals_bk, srccoeij_vx[s], srccoeval_vx[s], ρ_ihalf, it, s, dt)
        @parallel (1:nsrcpts_vz) inject_external_sources2D_vz!(adjvz, residuals_bk, srccoeij_vz[s], srccoeval_vz[s], ρ_jhalf, it, s, dt)
    end

    return
end