@views function forward!(
    ::AcousticWaveEquation,
    ::ReflectiveBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, backend
)
    nx = model.nx
    nt = model.nt
    dx = model.dx
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx)
    pcur = backend.zeros(nx)
    pnew = backend.zeros(nx)
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
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*it, maximum(abs.(Array(pcur))))
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(model.snapshots[:, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end

@views function forward!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel1D, possrcs, posrecs, srctf, traces, backend
)
    # Numerics
    nt = model.nt
    nx = model.nx
    dx = model.dx
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx)
    pcur = backend.zeros(nx)
    pnew = backend.zeros(nx)
    fact_a = backend.Data.Array( model.fact )
    # Initialize CPML arrays
    ψ_l = backend.zeros(halo+1)
    ψ_r = backend.zeros(halo+1)
    ξ_l = backend.zeros(halo)
    ξ_r = backend.zeros(halo)
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
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*it, maximum(abs.(Array(pcur))))
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(model.snapshots[:, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end

@views function forward!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel2D, possrcs, posrecs, srctf, traces, backend
)
    # Numerics
    nt = model.nt
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx, ny)
    pcur = backend.zeros(nx, ny)
    pnew = backend.zeros(nx, ny)
    fact_a = backend.Data.Array( model.fact )
    # Initialize CPML arrays
    ψ_x_l = backend.zeros(halo+1, ny)
    ψ_x_r = backend.zeros(halo+1, ny)
    ξ_x_l = backend.zeros(halo, ny)
    ξ_x_r = backend.zeros(halo, ny)
    ψ_y_l = backend.zeros(nx, halo+1)
    ψ_y_r = backend.zeros(nx, halo+1)
    ξ_y_l = backend.zeros(nx, halo)
    ξ_y_r = backend.zeros(nx, halo)
    # Wrap CPML coefficient arrays
    a_x_l = backend.Data.Array( model.cpmlcoeffs_x.a_l )
    a_x_r = backend.Data.Array( model.cpmlcoeffs_x.a_r )
    a_x_hl = backend.Data.Array( model.cpmlcoeffs_x.a_hl )
    a_x_hr = backend.Data.Array( model.cpmlcoeffs_x.a_hr )
    b_K_x_l = backend.Data.Array( model.cpmlcoeffs_x.b_K_l )
    b_K_x_r = backend.Data.Array( model.cpmlcoeffs_x.b_K_r )
    b_K_x_hl = backend.Data.Array( model.cpmlcoeffs_x.b_K_hl )
    b_K_x_hr = backend.Data.Array( model.cpmlcoeffs_x.b_K_hr )
    a_y_l = backend.Data.Array( model.cpmlcoeffs_y.a_l )
    a_y_r = backend.Data.Array( model.cpmlcoeffs_y.a_r )
    a_y_hl = backend.Data.Array( model.cpmlcoeffs_y.a_hl )
    a_y_hr = backend.Data.Array( model.cpmlcoeffs_y.a_hr )
    b_K_y_l = backend.Data.Array( model.cpmlcoeffs_y.b_K_l )
    b_K_y_r = backend.Data.Array( model.cpmlcoeffs_y.b_K_r )
    b_K_y_hl = backend.Data.Array( model.cpmlcoeffs_y.b_K_hl )
    b_K_y_hr = backend.Data.Array( model.cpmlcoeffs_y.b_K_hr )
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )

    # Time loop
    for it = 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, dx, dy,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*it, maximum(abs.(Array(pcur))))
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(model.snapshots[:, :, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end

@views function forward!(
    ::AcousticWaveEquation,
    ::CPMLBoundaryCondition,
    model::WaveModel3D, possrcs, posrecs, srctf, traces, backend
)
    # Numerics
    nt = model.nt
    nx = model.nx
    ny = model.ny
    nz = model.nz
    dx = model.dx
    dy = model.dy
    dz = model.dz
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(nx, ny, nz)
    pcur = backend.zeros(nx, ny, nz)
    pnew = backend.zeros(nx, ny, nz)
    fact_a = backend.Data.Array( model.fact )
    # Initialize CPML arrays
    ψ_x_l = backend.zeros(halo+1, ny, nz)
    ψ_x_r = backend.zeros(halo+1, ny, nz)
    ξ_x_l = backend.zeros(halo, ny, nz)
    ξ_x_r = backend.zeros(halo, ny, nz)
    ψ_y_l = backend.zeros(nx, halo+1, nz)
    ψ_y_r = backend.zeros(nx, halo+1, nz)
    ξ_y_l = backend.zeros(nx, halo, nz)
    ξ_y_r = backend.zeros(nx, halo, nz)
    ψ_z_l = backend.zeros(nx, ny, halo+1)
    ψ_z_r = backend.zeros(nx, ny, halo+1)
    ξ_z_l = backend.zeros(nx, ny, halo)
    ξ_z_r = backend.zeros(nx, ny, halo)
    # Wrap CPML coefficient arrays
    a_x_l = backend.Data.Array( model.cpmlcoeffs_x.a_l )
    a_x_r = backend.Data.Array( model.cpmlcoeffs_x.a_r )
    a_x_hl = backend.Data.Array( model.cpmlcoeffs_x.a_hl )
    a_x_hr = backend.Data.Array( model.cpmlcoeffs_x.a_hr )
    b_K_x_l = backend.Data.Array( model.cpmlcoeffs_x.b_K_l )
    b_K_x_r = backend.Data.Array( model.cpmlcoeffs_x.b_K_r )
    b_K_x_hl = backend.Data.Array( model.cpmlcoeffs_x.b_K_hl )
    b_K_x_hr = backend.Data.Array( model.cpmlcoeffs_x.b_K_hr )
    a_y_l = backend.Data.Array( model.cpmlcoeffs_y.a_l )
    a_y_r = backend.Data.Array( model.cpmlcoeffs_y.a_r )
    a_y_hl = backend.Data.Array( model.cpmlcoeffs_y.a_hl )
    a_y_hr = backend.Data.Array( model.cpmlcoeffs_y.a_hr )
    b_K_y_l = backend.Data.Array( model.cpmlcoeffs_y.b_K_l )
    b_K_y_r = backend.Data.Array( model.cpmlcoeffs_y.b_K_r )
    b_K_y_hl = backend.Data.Array( model.cpmlcoeffs_y.b_K_hl )
    b_K_y_hr = backend.Data.Array( model.cpmlcoeffs_y.b_K_hr )
    a_z_l = backend.Data.Array( model.cpmlcoeffs_z.a_l )
    a_z_r = backend.Data.Array( model.cpmlcoeffs_z.a_r )
    a_z_hl = backend.Data.Array( model.cpmlcoeffs_z.a_hl )
    a_z_hr = backend.Data.Array( model.cpmlcoeffs_z.a_hr )
    b_K_z_l = backend.Data.Array( model.cpmlcoeffs_z.b_K_l )
    b_K_z_r = backend.Data.Array( model.cpmlcoeffs_z.b_K_r )
    b_K_z_hl = backend.Data.Array( model.cpmlcoeffs_z.b_K_hl )
    b_K_z_hr = backend.Data.Array( model.cpmlcoeffs_z.b_K_hr )
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )

    # Time loop
    for it = 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, dx, dy, dz,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r, ψ_z_l, ψ_z_r, ξ_z_l, ξ_z_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            a_z_hl, a_z_hr, b_K_z_hl, b_K_z_hr,
            a_z_l, a_z_r, b_K_z_l, b_K_z_r,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*it, maximum(abs.(Array(pcur))))
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array(pcur))))
            copyto!(model.snapshots[:, :, :, div(it, model.snapevery)], pcur)
        end
    end

    # Save traces
    copyto!(traces, Array( traces_a ))
end