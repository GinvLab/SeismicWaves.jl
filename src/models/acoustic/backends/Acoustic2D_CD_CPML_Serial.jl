
module Acoustic2D_CD_CPML_Serial

using SeismicWaves.FiniteDifferencesMacros
using SeismicWaves.FDGeneratedFunctions

include("shared/smooth_gradient_2D.jl")

# Dummy data module
module Data
Array = Base.Array
end

# the backend needs these
ones = Base.ones
zeros = Base.zeros

function update_ψ_x!(
    pcur, _dx, ny, halo, ψ_x, b_x_half, a_x_half
)
    for j in 1:ny
        for i in 1:2halo
            # Shift index to right side if beyond left boundary
            ii = i > halo ? size(pcur, 1) - halo - 1 + (i - halo) : i
            # Update CPML memory variable
            @∂̃x(pcur, a_x_half, b_x_half, ψ_x,
                order=2, I=(ii,j), _Δ=_dx,
                halo=halo, halfgrid=true)
        end
    end
end

function update_ψ_y!(
    pcur, _dy, nx, halo, ψ_y, b_y_half, a_y_half
)
    for j in 1:2halo
        for i in 1:nx
            # Shift index to right side if beyond left boundary
            jj = j > halo ? size(pcur, 2) - halo - 1 + (j - halo) : j
            # Update CPML memory variable
            @∂̃y(pcur, a_y_half, b_y_half, ψ_y,
                order=2, I=(i,jj), _Δ=_dy,
                halo=halo, halfgrid=true)
        end
    end
end

function update_p_CPML!(
    pold, pcur, pnew, fact, _dx, _dy, nx, ny,
    halo, ψ_x, ψ_y, ξ_x, ξ_y, b_x, b_y, a_x, a_y
)

    ##########################
    # ∂²p/∂t² = c² * ∇²p     #
    # fact = c² * dt²        #
    ##########################

    for j in 2:(ny-1)
        for i in 2:(nx-1)
            # Compute pressure Laplacian
            ∇²p = @∇̃²(pcur, a_x, b_x, ψ_x, ξ_x, a_y, b_y, ψ_y, ξ_y,
                    order=2, I=(i,j), _Δ=(_dx, _dy),
                    halo=halo)

            # update pressure
            pnew[i, j] = 2.0 * pcur[i, j] - pold[i, j] + fact[i, j] * ∇²p
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

prescale_residuals!(residuals, posrecs, fact) =
    for ir in axes(posrecs, 1)
        irec = floor(Int, posrecs[ir, 1])
        jrec = floor(Int, posrecs[ir, 2])
        for it in axes(residuals, 1) # nt
            residuals[it, ir] *= fact[irec, jrec]
        end
    end

function forward_onestep_CPML!(
    model, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    grid = model.grid
    nx, ny = grid.size
    dx, dy = grid.spacing
    pold, pcur, pnew = grid.fields["pold"].value, grid.fields["pcur"].value, grid.fields["pnew"].value
    fact = grid.fields["fact"].value
    ψ_x, ψ_y = grid.fields["ψ"].value
    ξ_x, ξ_y = grid.fields["ξ"].value
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    a_y = model.cpmlcoeffs[2].a
    a_y_half = model.cpmlcoeffs[2].a_h
    b_y = model.cpmlcoeffs[2].b
    b_y_half = model.cpmlcoeffs[2].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx
    _dy = 1 / dy

    # update ψ arrays
    update_ψ_x!(pcur, _dx, ny, halo, ψ_x, b_x_half, a_x_half)
    update_ψ_y!(pcur, _dy, nx, halo, ψ_y, b_y_half, a_y_half)
    # update pressure and ξ arrays
    update_p_CPML!(
        pold, pcur, pnew, fact, _dx, _dy, nx, ny,
        halo, ψ_x, ψ_y, ξ_x, ξ_y, b_x, b_y, a_x, a_y
    )
    # inject sources
    inject_sources!(pnew, dt2srctf, possrcs, it)
    # record receivers
    if save_trace
        record_receivers!(pnew, traces, posrecs, it)
    end

    # Exchange pressures in grid
    grid.fields["pold"] = grid.fields["pcur"]
    grid.fields["pcur"] = grid.fields["pnew"]
    grid.fields["pnew"] = grid.fields["pold"]
    return nothing
end

function adjoint_onestep_CPML!(model, possrcs, dt2srctf, it)
    # Extract info from grid
    grid = model.grid
    nx, ny = grid.size
    dx, dy = grid.spacing
    pold, pcur, pnew = grid.fields["adjold"].value, grid.fields["adjcur"].value, grid.fields["adjnew"].value
    fact = grid.fields["fact"].value
    ψ_x, ψ_y = grid.fields["ψ_adj"].value
    ξ_x, ξ_y = grid.fields["ξ_adj"].value
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    a_y = model.cpmlcoeffs[2].a
    a_y_half = model.cpmlcoeffs[2].a_h
    b_y = model.cpmlcoeffs[2].b
    b_y_half = model.cpmlcoeffs[2].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx = 1 / dx
    _dy = 1 / dy

    # update ψ arrays
    update_ψ_x!(pcur, _dx, ny, halo, ψ_x, b_x_half, a_x_half)
    update_ψ_y!(pcur, _dy, nx, halo, ψ_y, b_y_half, a_y_half)
    # update pressure and ξ arrays
    update_p_CPML!(
        pold, pcur, pnew, fact, _dx, _dy, nx, ny,
        halo, ψ_x, ψ_y, ξ_x, ξ_y, b_x, b_y, a_x, a_y
    )
    # inject sources
    inject_sources!(pnew, dt2srctf, possrcs, it)

    # Exchange pressures in grid
    grid.fields["adjold"] = grid.fields["adjcur"]
    grid.fields["adjcur"] = grid.fields["adjnew"]
    grid.fields["adjnew"] = grid.fields["adjold"]
    return nothing
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
