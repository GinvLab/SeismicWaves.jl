@parallel_indices (i, j) function update_ux!(
    uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
    halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
    freetop
)
    # Compute partial derivatives
    ∂σxx∂x = ∂̃x4th(σxx, a_x_half, b_x_half, ψ_∂σxx∂x, (i, j), _Δx, halo; half=false)
    # if free surface, mirror the top boundary
    ∂σxz∂z = ∂̃y4th(σxz, a_z, b_z, ψ_∂σxz∂z, (i, j-1), _Δz, halo; half=true, mlb=freetop)
    # Update displacement
    uxnew[i, j] = 2*uxcur[i, j] - uxold[i, j] + Δt^2 / ρ_ihalf[i,j] * (∂σxx∂x + ∂σxz∂z)

    return nothing
end

@parallel_indices (i, j) function update_uz!(
    uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
    halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
    freetop
)
    # Compute partial derivatives
    ∂σxz∂x = ∂̃x4th(σxz, a_x, b_x, ψ_∂σxz∂x, (i-1, j), _Δx, halo; half=true)
    # if free surface, mirror the top boundary
    ∂σzz∂z = ∂̃y4th(σzz, a_z_half, b_z_half, ψ_∂σzz∂z, (i, j), _Δz, halo; half=false, mlb=freetop)
    # Update displacement
    uznew[i, j] = 2*uzcur[i, j] - uzold[i, j] + Δt^2 / ρ_jhalf[i,j] * (∂σzz∂z + ∂σxz∂x)

    return nothing
end

@parallel_indices (i, j) function update_σxx_σzz!(
    σxx, σzz, ux, uz, λ, μ, _Δx, _Δz,
    halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
    freetop
)
    # Compute partial derivatives
    ∂ux∂x = ∂̃x4th(ux, a_x, b_x, ψ_∂ux∂x, (i-1, j), _Δx, halo; half=true)
    ∂uz∂z = ∂̃y4th(uz, a_z, b_z, ψ_∂uz∂z, (i, j-1), _Δz, halo; half=true, mlb=freetop)
    # Update normal stress
    σxx[i, j] = (λ[i,j] + 2*μ[i,j]) * ∂ux∂x + λ[i,j] * ∂uz∂z
    σzz[i, j] = λ[i,j] * ∂ux∂x + (λ[i,j] + 2*μ[i,j]) * ∂uz∂z

    return nothing
end

@parallel_indices (i, j) function update_σxz!(
    σxz, ux, uz, μ_ihalf_jhalf, _Δx, _Δz,
    halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
    freetop
)
    # Compute partial derivatives
    ∂ux∂z = ∂̃y4th(ux, a_z_half, b_z_half, ψ_∂ux∂z, (i, j), _Δz, halo; half=false, mlb=freetop)
    ∂uz∂x = ∂̃x4th(uz, a_x_half, b_x_half, ψ_∂uz∂x, (i, j), _Δx, halo; half=false)
    # Update shear stress
    σxz[i, j] = μ_ihalf_jhalf[i,j] * (∂ux∂z + ∂uz∂x)

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

@parallel_indices (p) function inject_momten_sources2D_onecomp!(σ, srctf_bk, srccoeij_bk, srccoeval_bk, it)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    σ[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it]

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

@parallel_indices (p) function inject_external_sources2D_onecomp!(uu, srctf_bk, srccoeij_bk, srccoeval_bk, ρ_half, it, s, d, dt)
    isrc, jsrc = srccoeij_bk[p, 1], srccoeij_bk[p, 2]
    uu[isrc, jsrc] += srccoeval_bk[p] * srctf_bk[it, d, s] / ρ_half[isrc, jsrc] * dt^2
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

@parallel_indices (p) function record_stress2D_σxx_σzz!(σxx, σzz, traces_xx_bk_buf, traces_zz_bk_buf, reccoeij_xx, reccoeval_xx)
    irec, jrec = reccoeij_xx[p, 1], reccoeij_xx[p, 2]
    traces_xx_bk_buf[p] = reccoeval_xx[p] * σxx[irec, jrec]
    traces_zz_bk_buf[p] = reccoeval_xx[p] * σzz[irec, jrec]
    return nothing
end

@parallel_indices (p) function record_stress2D_σxz!(σxz, traces_xz_bk_buf, reccoeij_xz, reccoeval_xz)
    irec, jrec = reccoeij_xz[p, 1], reccoeij_xz[p, 2]
    traces_xz_bk_buf[p] = reccoeval_xz[p] * σxz[irec, jrec]
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
    freetop = model.cpmlparams.freeboundtop
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
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freetop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freetop
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
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freetop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freetop
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
    freetop = model.cpmlparams.freeboundtop
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
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freetop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freetop
    )

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freetop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freetop
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
    freetop = model.cpmlparams.freeboundtop
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
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freetop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freetop
    )

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freetop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freetop
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

function forward_onestep_ccout_CPML!(
    model,
    refreccoeij,
    refreccoeval,
    srccoeij_xx,
    srccoeval_xx,
    srccoeij_xz,
    srccoeval_xz,
    recrefscal_srctf_bk,
    reduced_buf,
    traces_xx_bk_buf,
    traces_zz_bk_buf,
    traces_xz_bk_buf,
    traces_xx_bk,
    traces_zz_bk,
    traces_xz_bk,
    it;
    d=1,
    save_trace=true
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
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

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freetop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freetop
    )

    # Inject reference receiver as source (external force type)
    nsrcpts = size(refreccoeij, 1)
    if d == 1
        @parallel (1:nsrcpts) inject_external_sources2D_onecomp!(uxnew, recrefscal_srctf_bk, refreccoeij, refreccoeval, ρ_ihalf, it, 1, d, Δt)
    elseif d == 2
        @parallel (1:nsrcpts) inject_external_sources2D_onecomp!(uznew, recrefscal_srctf_bk, refreccoeij, refreccoeval, ρ_jhalf, it, 1, d, Δt)
    end

    # Update normal stress
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_σxx_σzz!(
        σxx, σzz, uxnew, uznew, λ, μ, _Δx, _Δz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freetop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxnew, uznew, μ_ihalf_jhalf, _Δx, _Δz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freetop
    )

    # Record moment tensor
    if save_trace
        nrecs = size(srccoeij_xx, 1)
        for r in 1:nrecs
            nrecspts_xx = size(srccoeij_xx[r], 1)
            nrecspts_xz = size(srccoeij_xz[r], 1)

            @parallel (1:nrecspts_xx) record_stress2D_σxx_σzz!(σxx, σzz, traces_xx_bk_buf[r], traces_zz_bk_buf[r], srccoeij_xx[r], srccoeval_xx[r])
            @parallel (1:nrecspts_xz) record_stress2D_σxz!(σxz, traces_xz_bk_buf[r], srccoeij_xz[r], srccoeval_xz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_xx_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_zz_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[3][r:r]), traces_xz_bk_buf[r])
        end
        copyto!(@view(traces_xx_bk[it, :]), reduced_buf[1])
        copyto!(@view(traces_zz_bk[it, :]), reduced_buf[2])
        copyto!(@view(traces_xz_bk[it, :]), reduced_buf[3])
        reduced_buf[1] .= 0
        reduced_buf[2] .= 0
        reduced_buf[3] .= 0
    end

    # Swap old and current displacements
    grid.fields["uold"] = grid.fields["ucur"]
    grid.fields["ucur"] = grid.fields["unew"]
    grid.fields["unew"] = grid.fields["uold"]

    return nothing
end

function forward_onestep_ccin_CPML!(
    model,
    srccoeij_xx,
    srccoeval_xx,
    srccoeij_xz,
    srccoeval_xz,
    reccoeij_ux,
    reccoeval_ux,
    reccoeij_uz,
    reccoeval_uz,
    srctf_Mxx_bk,
    srctf_Mzz_bk,
    srctf_Mxz_bk,
    reduced_buf,
    traces_ux_bk_buf,
    traces_uz_bk_buf,
    traces_bk,
    it::Int,
    nt::Int;
    save_trace::Bool=true
)
    # Extract info from grid
    freetop = model.cpmlparams.freeboundtop
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
    idxzσxx = freetop ? (1:nz-1) : (2:nz-1)
    @parallel (2:nx-1, idxzσxx) update_σxx_σzz!(
        σxx, σzz, uxcur, uzcur, λ, μ, _Δx, _Δz,
        halo, a_x, a_z, b_x, b_z, ψ_∂ux∂x, ψ_∂uz∂z,
        freetop
    )
    # Update shear stress
    @parallel (1:nx-1, 1:nz-1) update_σxz!(
        σxz, uxcur, uzcur, μ_ihalf_jhalf, _Δx, _Δz,
        halo, a_x_half, a_z_half, b_x_half, b_z_half, ψ_∂ux∂z, ψ_∂uz∂x,
        freetop
    )

    # Inject sources (moment tensor type of internal force)
    if it >= 1
        nsrcs = size(srccoeij_xx, 1)
        for s in 1:nsrcs
            nsrcpts_xx = size(srccoeij_xx[s], 1)
            nsrcpts_xz = size(srccoeij_xz[s], 1)
            @parallel (1:nsrcpts_xx) inject_momten_sources2D_onecomp!(σxx, @view(srctf_Mxx_bk[:, s]), srccoeij_xx[s], srccoeval_xx[s], it)
            @parallel (1:nsrcpts_xx) inject_momten_sources2D_onecomp!(σzz, @view(srctf_Mzz_bk[:, s]), srccoeij_xx[s], srccoeval_xx[s], it)
            @parallel (1:nsrcpts_xz) inject_momten_sources2D_onecomp!(σxz, @view(srctf_Mxz_bk[:, s]), srccoeij_xz[s], srccoeval_xz[s], it)
        end
    end

    # Update displacements
    @parallel (1:nx-1, 1:nz) update_ux!(
        uxnew, uxcur, uxold, σxx, σxz, ρ_ihalf, _Δx, _Δz, Δt,
        halo, a_x_half, a_z, b_x_half, b_z, ψ_∂σxx∂x, ψ_∂σxz∂z,
        freetop
    )
    @parallel (1:nx, 1:nz-1) update_uz!(
        uznew, uzcur, uzold, σxz, σzz, ρ_jhalf, _Δx, _Δz, Δt,
        halo, a_x, a_z_half, b_x, b_z_half, ψ_∂σxz∂x, ψ_∂σzz∂z,
        freetop
    )

    # Record receivers
    if save_trace
        nrecs = size(reccoeij_ux, 1)
        for r in 1:nrecs
            nrecspts_ux = size(reccoeij_ux[r], 1)
            nrecspts_uz = size(reccoeij_uz[r], 1)
            @parallel (1:nrecspts_ux) record_receivers2D_ux!(uxcur, traces_ux_bk_buf[r], reccoeij_ux[r], reccoeval_ux[r])
            @parallel (1:nrecspts_uz) record_receivers2D_uz!(uzcur, traces_uz_bk_buf[r], reccoeij_uz[r], reccoeval_uz[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[1][r:r]), traces_ux_bk_buf[r])
            Base.mapreducedim!(identity, +, @view(reduced_buf[2][r:r]), traces_uz_bk_buf[r])
        end
        copyto!(@view(traces_bk[it + nt + 1, 1, :]), reduced_buf[1])
        copyto!(@view(traces_bk[it + nt + 1, 2, :]), reduced_buf[2])
        reduced_buf[1] .= 0
        reduced_buf[2] .= 0
    end

    # Swap old and current displacements
    grid.fields["uold"] = grid.fields["ucur"]
    grid.fields["ucur"] = grid.fields["unew"]
    grid.fields["unew"] = grid.fields["uold"]

    return nothing
end