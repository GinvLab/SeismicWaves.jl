function forward!(
    ::AcousticWaveEquation,
    ::ReflectiveBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, backend
)
    nx = model.nx
    nt = model.nt
    dx = model.dx
    # Initialize pressure and factors arrays
    pold = backend.Data.Array( zeros(nx) )
    pcur = backend.Data.Array( zeros(nx) )
    pnew = backend.Data.Array( zeros(nx) )
    fact_a = backend.Data.Array( model.fact )
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )

    # Time loop
    for it = 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep!(
            pold, pcur, pnew, fact_a, dx,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            copyto!(model.snapshots[:, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end

function forward!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, backend
)
    nx = model.nx
    nt = model.nt
    dx = model.dx
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.Data.Array( zeros(nx) )
    pcur = backend.Data.Array( zeros(nx) )
    pnew = backend.Data.Array( zeros(nx) )
    fact_a = backend.Data.Array( model.fact )
    # Initialize CPML arrays
    ψ_l = backend.Data.Array( zeros(halo+1) )
    ψ_r = backend.Data.Array( zeros(halo+1) )
    ξ_l = backend.Data.Array( zeros(halo) )
    ξ_r = backend.Data.Array( zeros(halo) )
    # Wrap CPML coefficient arrays
    a_x_l = backend.Data.Array( model.cpmlcoeffs.a_l )
    a_x_r = backend.Data.Array( model.cpmlcoeffs.a_r )
    a_x_hl = backend.Data.Array( model.cpmlcoeffs.a_hl )
    a_x_hr = backend.Data.Array( model.cpmlcoeffs.a_hr )
    b_K_x_l = backend.Data.Array( model.cpmlcoeffs.b_K_l )
    b_K_x_r = backend.Data.Array( model.cpmlcoeffs.b_K_r )
    b_K_x_hl = backend.Data.Array( model.cpmlcoeffs.b_K_hl )
    b_K_x_hr = backend.Data.Array( model.cpmlcoeffs.b_K_hr )
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )

    # Time loop
    for it = 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, dx,
            halo, ψ_l, ψ_r, ξ_l, ξ_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            copyto!(model.snapshots[:, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end