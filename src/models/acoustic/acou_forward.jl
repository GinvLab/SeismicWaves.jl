
@views function swforward_1shot!(
    model::Acoustic_CD_CPML_WaveSimul, # includes types Acoustic_CD_CPML_WaveSimul{1}, ...{2} and ...{3}
    backend::Module,
    possrcs, posrecs, srctf, traces
    # ::Acoustic_CD_WaveSimul,
    # ::CPMLBoundaryCondition,
    # backend::Module,
    # model::WaveSimul, possrcs, posrecs, srctf, traces
    )
    
    # Numerics
    N = length(model.ns)
    nt = model.nt
    halo = model.halo
    # Initialize pressure and factors arrays
    pold = backend.zeros(model.ns...)
    pcur = backend.zeros(model.ns...)
    pnew = backend.zeros(model.ns...)
    fact_a = backend.Data.Array( model.fact )
    # Initialize CPML arrays
    ψ = []
    ξ = []
    for i = 1:N
        ψ_ns = [model.ns...]
        ξ_ns = [model.ns...]
        ψ_ns[i] = halo+1
        ξ_ns[i] = halo
        append!(ψ, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
        append!(ξ, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
    end
    # Wrap CPML coefficient arrays
    a_coeffs = []
    b_K_coeffs = []
    for i = 1:N
        append!(a_coeffs, backend.Data.Array.([
            model.cpmlcoeffs[i].a_l,
            model.cpmlcoeffs[i].a_r,
            model.cpmlcoeffs[i].a_hl,
            model.cpmlcoeffs[i].a_hr
        ]))
        append!(b_K_coeffs, backend.Data.Array.([
            model.cpmlcoeffs[i].b_K_l,
            model.cpmlcoeffs[i].b_K_r,
            model.cpmlcoeffs[i].b_K_hl,
            model.cpmlcoeffs[i].b_K_hr
        ]))
    end
    # Wrap sources and receivers arrays
    possrcs_a = backend.Data.Array( possrcs )
    posrecs_a = backend.Data.Array( posrecs )
    srctf_a = backend.Data.Array( srctf )
    traces_a = backend.Data.Array( traces )

    # Time loop
    for it = 1:nt
        # Compute one forward step
        pold, pcur, pnew = backend.forward_onestep_CPML!(
            pold, pcur, pnew, fact_a, model.gridspacing..., halo,
            ψ..., ξ..., a_coeffs..., b_K_coeffs...,
            possrcs_a, srctf_a, posrecs_a, traces_a, it
        )
        # Print timestep info
        if it % model.infoevery == 0
            @debug @sprintf("Iteration: %d, simulation time: %g [s], maximum absolute pressure: %g [Pa]", it, model.dt*(it-1), maximum(abs.(Array( pcur ))))
        end

        # Save snapshot
        if snapenabled(model) && it % model.snapevery == 0
            @debug @sprintf("Snapping iteration: %d, max absolute pressure: %g [Pa]", it, maximum(abs.(Array( pcur ))))
            model.snapshots[fill(Colon(), N)..., div(it, model.snapevery)] .= Array( pcur )
        end
    end

    # Save traces
    traces .= Array( traces_a )
    return
end
