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