module Acoustic1D_CD_CPML_Serial

using SeismicWaves.FiniteDifferencesMacros

# Dummy data module
module Data
Array = Base.Array
end

#####
ones = Base.ones
zeros = Base.zeros

function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i in 1:nsrcs
        isrc = floor(Int, possrcs[i, 1])
        pnew[isrc] += dt2srctf[it, i]
    end
end

function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s in 1:nrecs
        irec = floor(Int, posrecs[s, 1])
        traces[it, s] = pnew[irec]
    end
end

function update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    for i in 1:2halo
        # Shift index to right side if beyond left boundary
        ii = i > halo ? size(pcur, 1) - halo - 1 + (i - halo) : i
        # Update CPML memory variable
        @∂̃x(pcur, a_x_half, b_x_half, ψ_x,
            order=2, I=(ii,), _Δ=_dx,
            halo=halo, halfgrid=true)
    end
end

function update_p_CPML!(
    pold, pcur, pnew, fact, _dx, nx,
    halo, ψ_x, ξ_x, b_x, a_x
)
    ##########################
    # ∂²p/∂t² = c² * ∇²p     #
    # fact = c² * dt²        #
    ##########################
    
    for i in 2:(nx-1)
        # Compute pressure Laplacian
        ∇²p = @∇̃²(pcur, a_x, b_x, ψ_x, ξ_x,
                  order=2, I=(i,), _Δ=(_dx,),
                  halo=halo)
        # Update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * ∇²p
    end
end

prescale_residuals!(residuals, posrecs, fact) =
    for ir in axes(posrecs, 1)
        irec = floor(Int, posrecs[ir, 1])
        for it in axes(residuals, 1) # nt
            residuals[it, ir] *= fact[irec]
        end
    end

function forward_onestep_CPML!(
    model, possrcs, dt2srctf, posrecs, traces, it;
    save_trace=true
)
    # Extract info from grid
    grid = model.grid
    nx = grid.size[1]
    dx = grid.spacing[1]
    pold, pcur, pnew = grid.fields["pold"].value, grid.fields["pcur"].value, grid.fields["pnew"].value
    fact = grid.fields["fact"].value
    ψ_x = grid.fields["ψ"].value[1]
    ξ_x = grid.fields["ξ"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx  = 1 / dx

    # update ψ arrays
    update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    # update pressure and ξ arrays
    update_p_CPML!(
        pold, pcur, pnew, fact, _dx, nx,
        halo, ψ_x, ξ_x, b_x, a_x
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
    nx = grid.size[1]
    dx = grid.spacing[1]
    pold, pcur, pnew = grid.fields["adjold"].value, grid.fields["adjcur"].value, grid.fields["adjnew"].value
    fact = grid.fields["fact"].value
    ψ_x = grid.fields["ψ_adj"].value[1]
    ξ_x = grid.fields["ξ_adj"].value[1]
    a_x = model.cpmlcoeffs[1].a
    a_x_half = model.cpmlcoeffs[1].a_h
    b_x = model.cpmlcoeffs[1].b
    b_x_half = model.cpmlcoeffs[1].b_h
    halo = model.cpmlparams.halo
    # Precompute divisions
    _dx  = 1 / dx

    # update ψ arrays
    update_ψ!(pcur, _dx, halo, ψ_x, b_x_half, a_x_half)
    # update pressure and ξ arrays
    update_p_CPML!(
        pold, pcur, pnew, fact, _dx, nx,
        halo, ψ_x, ξ_x, b_x, a_x
    )
    # inject sources
    inject_sources!(pnew, dt2srctf, possrcs, it)

    # Exchange pressures in grid
    grid.fields["adjold"] = grid.fields["adjcur"]
    grid.fields["adjcur"] = grid.fields["adjnew"]
    grid.fields["adjnew"] = grid.fields["adjold"]

    return nothing
end

function correlate_gradient!(curgrad, adjcur, pcur, pold, pveryold, dt)
    _dt2 = 1 / dt^2
    nx = size(curgrad, 1)
    for i in 1:nx
        curgrad[i] = curgrad[i] + (adjcur[i] * (pcur[i] - 2.0 * pold[i] + pveryold[i]) * _dt2)
    end
end

###################
end # module
###################
