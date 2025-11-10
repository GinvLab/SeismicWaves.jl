@parallel_indices (i, j) function update_ux!(
    uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt, nx, nz,
    halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
    freeboundtop
)
    # Compute partial derivatives
    ∂σxx∂x = ∂σxx∂x_4th(σxx, i, j, _Δx, nx)
    ∂σxz∂z = ∂σxz∂z_4th(σxz, i, j, _Δz, nz, freeboundtop)
    
    # Add CPML attenuation to partial derivatives
    ∂σxx∂x_cpml = ∂̃x4th(σxx, ∂σxx∂x, a_x_half, b_x_half, ψ_∂σxx∂x, (i, j)  , _Δx, halo; half=false)
    ∂σxz∂z_cpml = ∂̃y4th(σxz, ∂σxz∂z, a_z     , b_z     , ψ_∂σxz∂z, (i, j-1), _Δz, halo; half=true)

    # Update displacement using 2nd-order time scheme
    uxnew[i, j] = 2*uxcur[i, j] - uxold[i, j] + Δt^2 / ρ_ihalf[i,j] * (∂σxx∂x_cpml + ∂σxz∂z_cpml)

    return nothing
end

@parallel_indices (i, j) function update_uz!(
    uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt, nx, nz,
    halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
    freeboundtop
)
    # Compute partial derivatives
    ∂σxz∂x = ∂σxz∂x_4th(σxz, i, j, _Δx, nx)
    ∂σzz∂z = ∂σzz∂z_4th(σzz, i, j, _Δz, nz, freeboundtop)

    # Add CPML attenuation to partial derivatives
    ∂σxz∂x_cpml = ∂̃x4th(σxz, ∂σxz∂x, a_x     , b_x     , ψ_∂σxz∂x, (i-1, j), _Δx, halo; half=true)
    ∂σzz∂z_cpml = ∂̃y4th(σzz, ∂σzz∂z, a_z_half, b_z_half, ψ_∂σzz∂z, (i, j)  , _Δz, halo; half=false)

    # Update displacement
    uznew[i, j] = 2*uzcur[i, j] - uzold[i, j] + Δt^2 / ρ_jhalf[i,j] * (∂σxz∂x_cpml + ∂σzz∂z_cpml)

    return nothing
end

@parallel_indices (i, j) function update_σxx_σzz!(
    σxx, σzz, ux, uz, λ, μ, _Δx, _Δz, nx, nz,
    halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
    freeboundtop
)
    # Compute partial derivatives
    ∂ux∂x = ∂ux∂x_4th(ux, i, j, _Δx, nx)
    ∂uz∂z = ∂uz∂z_4th(ux, uz, λ, μ, i, j, _Δx, _Δz, nx, nz, freeboundtop)
    # Add CPML attenuation to partial derivatives
    ∂ux∂x_cpml = ∂̃x4th(ux, ∂ux∂x, a_x, b_x, ψ_∂ux∂x, (i-1, j), _Δx, halo; half=true)
    ∂uz∂z_cpml = ∂̃y4th(uz, ∂uz∂z, a_z, b_z, ψ_∂uz∂z, (i, j-1), _Δz, halo; half=true)

    # Update normal stresses
    σxx[i, j] = (λ[i,j] + 2*μ[i,j]) * ∂ux∂x_cpml + λ[i,j] * ∂uz∂z_cpml
    if j == 1
        σzz[i, j] = 0
    else
        σzz[i, j] = λ[i,j] * ∂ux∂x_cpml + (λ[i,j] + 2*μ[i,j]) * ∂uz∂z_cpml
    end

    return nothing
end

@parallel_indices (i, j) function update_σxz!(
    σxz, ux, uz, μ_ihalf_jhalf, _Δx, _Δz, nx, nz,
    halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
    freeboundtop
)
    # Compute partial derivatives
    ∂uz∂x = ∂uz∂x_4th(uz, i, j, _Δx, nx)
    ∂ux∂z = ∂ux∂z_4th(ux, i, j, _Δz, nz, freeboundtop)

    # Add CPML attenuation to partial derivatives
    ∂uz∂x_cpml = ∂̃x4th(uz, ∂uz∂x, a_x_half, b_x_half, ψ_∂uz∂x, (i, j), _Δx, halo; half=false)
    ∂ux∂z_cpml = ∂̃y4th(ux, ∂ux∂z, a_z_half, b_z_half, ψ_∂ux∂z, (i, j), _Δz, halo; half=false)

    # Update shear stress
    σxz[i, j] = μ_ihalf_jhalf[i,j] * (∂uz∂x_cpml + ∂ux∂z_cpml)

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx, Mzz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxx[isrc, jsrc] += Mxx[s] * srccoeval_bk[p] * srctf_bk[it, s]
    σzz[isrc, jsrc] += Mzz[s] * srccoeval_bk[p] * srctf_bk[it, s]

    return nothing
end

@parallel_indices (p) function inject_momten_sources2D_σxz!(σxz, Mxz, srctf_bk, srccoeij_bk, srccoeval_bk, it, s)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σxz[isrc, jsrc] += Mxz[s] * srccoeval_bk[p] * srctf_bk[it, s]

    return nothing
end

@parallel_indices (p) function inject_external_sources2D_ux!(ux, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_ihalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    ux[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 1, s] / ρ_ihalf[isrc, jsrc] * dt^2
    return nothing
end

@parallel_indices (p) function inject_external_sources2D_uz!(uz, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_jhalf, it, s, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    uz[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, 2, s] / ρ_jhalf[isrc, jsrc] * dt^2
    return nothing
end

@parallel_indices (p) function record_receivers2D_ux!(ux, traces_ux_bk_buf, reccoeij_ux, reccoeval_ux)
    irec, jrec = reccoeij_ux[p, 1], reccoeij_ux[p, 2]
    traces_ux_bk_buf[p] = reccoeval_ux[p] * ux[irec, jrec]
    return nothing
end

@parallel_indices (p) function record_receivers2D_uz!(uz, traces_uz_bk_buf, reccoeij_uz, reccoeval_uz)
    irec, jrec = reccoeij_uz[p, 1], reccoeij_uz[p, 2]
    traces_uz_bk_buf[p] = reccoeval_uz[p] * uz[irec, jrec]
    return nothing
end

function forward_onestep_CPML!(
    model,
    srccoeij_xx,
    srccoeval_xx,
    srccoeij_xz,
    srccoeval_xz,
    reccoeij_ux,
    reccoeval_ux,
    reccoeij_uz,
    reccoeval_uz,
    srctf_bk,
    reduced_buf,
    traces_ux_bk_buf,
    traces_uz_bk_buf,
    traces_bk,
    it::Int,
    Mxx_bk,
    Mzz_bk,
    Mxz_bk;
    save_trace::Bool=true
)
    # Extract info from grid
    cpmlcoeffs = model.cpmlcoeffs
    Δx, Δz = model.grid.spacing

    Δt = model.dt
    nx, nz = model.grid.size
    halo = model.cpmlparams.halo
    grid = model.grid

    uxold, uzold = grid.fields["uold"].value
    uxcur, uzcur = grid.fields["ucur"].value
    uxnew, uznew = grid.fields["unew"].value
    σxx, σzz, σxz = grid.fields["σ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["ψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["ψ_∂σ∂z"].value
    ψ_∂ux∂x, ψ_∂uz∂x = grid.fields["ψ_∂u∂x"].value
    ψ_∂ux∂z, ψ_∂uz∂z = grid.fields["ψ_∂u∂z"].value

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
    _Δx = 1 / Δx
    _Δz = 1 / Δz

    # Update normal stress
    freeboundtop = model.cpmlparams.freeboundtop
    idxσxx = freeboundtop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz, nx, nz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freeboundtop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz, nx, nz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freeboundtop
    )

    # Inject sources (moment tensor type of internal force)
    nsrcs = size(srccoeij_xx, 1)
    for s in 1:nsrcs
        nsrcpts_xx = size(srccoeij_xx[s], 1)
        nsrcpts_xz = size(srccoeij_xz[s], 1)
        @parallel (1:nsrcpts_xx) inject_momten_sources2D_σxx_σzz!(σxx, σzz, Mxx_bk, Mzz_bk, srctf_bk, srccoeij_xx[s], srccoeval_xx[s], it, s)
        @parallel (1:nsrcpts_xz) inject_momten_sources2D_σxz!(σxz, Mxz_bk, srctf_bk, srccoeij_xz[s], srccoeval_xz[s], it, s)
    end

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freeboundtop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freeboundtop
    )

    # Record receivers
    if save_trace
        nrecs = size(reccoeij_ux, 1)
        for r in 1:nrecs
            nrecspts_ux = size(reccoeij_ux[r], 1)
            nrecspts_uz = size(reccoeij_uz[r], 1)
            @parallel (1:nrecspts_ux) record_receivers2D_ux!(uxnew, traces_ux_bk_buf[r], reccoeij_ux[r], reccoeval_ux[r])
            @parallel (1:nrecspts_uz) record_receivers2D_uz!(uznew, traces_uz_bk_buf[r], reccoeij_uz[r], reccoeval_uz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_ux_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_uz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf[1])
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf[2])
        reduced_buf[1] .= 0
        reduced_buf[2] .= 0
    end

    # Swap old and current displacements
    grid.fields["uold"] = grid.fields["ucur"]
    grid.fields["ucur"] = grid.fields["unew"]
    grid.fields["unew"] = grid.fields["uold"]

    return nothing
end

function forward_onestep_CPML!(
    model,
    srccoeij_ux,
    srccoeval_ux,
    srccoeij_uz,
    srccoeval_uz,
    reccoeij_ux,
    reccoeval_ux,
    reccoeij_uz,
    reccoeval_uz,
    srctf_bk,
    reduced_buf,
    traces_ux_bk_buf,
    traces_uz_bk_buf,
    traces_bk,
    it::Int;
    save_trace::Bool=true
)
    # Extract info from grid
    cpmlcoeffs = model.cpmlcoeffs
    Δx, Δz = model.grid.spacing

    Δt = model.dt
    nx, nz = model.grid.size
    halo = model.cpmlparams.halo
    grid = model.grid

    uxold, uzold = grid.fields["uold"].value
    uxcur, uzcur = grid.fields["ucur"].value
    uxnew, uznew = grid.fields["unew"].value
    σxx, σzz, σxz = grid.fields["σ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["ψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["ψ_∂σ∂z"].value
    ψ_∂ux∂x, ψ_∂uz∂x = grid.fields["ψ_∂u∂x"].value
    ψ_∂ux∂z, ψ_∂uz∂z = grid.fields["ψ_∂u∂z"].value

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
    _Δx = 1 / Δx
    _Δz = 1 / Δz

    # Update normal stress
    freeboundtop = model.cpmlparams.freeboundtop
    idxσxx = freeboundtop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz, nx, nz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freeboundtop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz, nx, nz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freeboundtop
    )

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freeboundtop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freeboundtop
    )

    # Inject sources (external force type)
    nsrcs = size(srccoeij_ux, 1)
    for s in 1:nsrcs
        nsrcpts_ux = size(srccoeij_ux[s], 1)
        nsrcpts_uz = size(srccoeij_uz[s], 1)
        @parallel (1:nsrcpts_ux) inject_external_sources2D_ux!(uxnew, srctf_bk, srccoeij_ux[s], srccoeval_ux[s], ρ_ihalf, it, s, Δt)
        @parallel (1:nsrcpts_uz) inject_external_sources2D_uz!(uznew, srctf_bk, srccoeij_uz[s], srccoeval_uz[s], ρ_jhalf, it, s, Δt)
    end

    # Record receivers
    if save_trace
        nrecs = size(reccoeij_ux, 1)
        for r in 1:nrecs
            nrecspts_ux = size(reccoeij_ux[r], 1)
            nrecspts_uz = size(reccoeij_uz[r], 1)
            @parallel (1:nrecspts_ux) record_receivers2D_ux!(uxnew, traces_ux_bk_buf[r], reccoeij_ux[r], reccoeval_ux[r])
            @parallel (1:nrecspts_uz) record_receivers2D_uz!(uznew, traces_uz_bk_buf[r], reccoeij_uz[r], reccoeval_uz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_ux_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_uz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it, 1, :]), reduced_buf[1])
        copyto!(@view(traces_bk[it, 2, :]), reduced_buf[2])
        reduced_buf[1] .= 0
        reduced_buf[2] .= 0
    end

    # Swap old and current displacements
    grid.fields["uold"] = grid.fields["ucur"]
    grid.fields["ucur"] = grid.fields["unew"]
    grid.fields["unew"] = grid.fields["uold"]

    return nothing
end

function adjoint_onestep_CPML!(
    model,
    reccoeij_ux,
    reccoeval_ux,
    reccoeij_uz,
    reccoeval_uz,
    residuals_bk,
    it::Int
)
    # Extract info from grid
    cpmlcoeffs = model.cpmlcoeffs
    Δx, Δz = model.grid.spacing
    Δt = model.dt
    nx, nz = model.grid.size
    halo = model.cpmlparams.halo
    grid = model.grid

    uxold, uzold = grid.fields["adjuold"].value
    uxcur, uzcur = grid.fields["adjucur"].value
    uxnew, uznew = grid.fields["adjunew"].value
    σxx, σzz, σxz = grid.fields["adjσ"].value

    ψ_∂σxx∂x, ψ_∂σxz∂x = grid.fields["adjψ_∂σ∂x"].value
    ψ_∂σzz∂z, ψ_∂σxz∂z = grid.fields["adjψ_∂σ∂z"].value
    ψ_∂ux∂x, ψ_∂uz∂x = grid.fields["adjψ_∂u∂x"].value
    ψ_∂ux∂z, ψ_∂uz∂z = grid.fields["adjψ_∂u∂z"].value

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
    _Δx = 1 / Δx
    _Δz = 1 / Δz

    # Update normal stress
    freeboundtop = model.cpmlparams.freeboundtop
    idxσxx = freeboundtop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz, nx, nz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freeboundtop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz, nx, nz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freeboundtop
    )

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freeboundtop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt, nx, nz,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freeboundtop
    )

    # Inject adjoint sources (external forces)
    nrecs = size(reccoeij_ux, 1)
    for r in 1:nrecs
        nrecspts_ux = size(reccoeij_ux[r], 1)
        nrecspts_uz = size(reccoeij_uz[r], 1)
        @parallel (1:nrecspts_ux) inject_external_sources2D_ux!(uxnew, residuals_bk, reccoeij_ux[r], reccoeval_ux[r], ρ_ihalf, it, r, Δt)
        @parallel (1:nrecspts_uz) inject_external_sources2D_uz!(uznew, residuals_bk, reccoeij_uz[r], reccoeval_uz[r], ρ_jhalf, it, r, Δt)
    end

    # Swap old and current displacements
    grid.fields["adjuold"] = grid.fields["adjucur"]
    grid.fields["adjucur"] = grid.fields["adjunew"]
    grid.fields["adjunew"] = grid.fields["adjuold"]

    return
end
