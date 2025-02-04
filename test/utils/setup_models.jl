using DSP, NumericalIntegration, LinearAlgebra

function gaussian_vel_1D(nx, c0, c0max, r, origin=(nx + 1) / 2)
    sigma = r / 3
    amp = c0max - c0
    f(x) = amp * exp(-(0.5 * (x - origin)^2 / sigma^2))
    return c0 .+ [f(x) for x in 1:nx]
end

function gaussian_vel_2D(nx, ny, c0, c0max, r, origin=[(nx + 1) / 2, (ny + 1) / 2])
    sigma = r / 3
    amp = c0max - c0
    f(x, y) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2) / sigma^2))
    return c0 .+ [f(x, y) for x in 1:nx, y in 1:ny]
end

function gaussian_vel_3D(nx, ny, nz, c0, c0max, r, origin=[(nx + 1) / 2, (ny + 1) / 2, (nz + 1) / 2])
    sigma = r / 3
    amp = c0max - c0
    f(x, y, z) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2 + (z - origin[3])^2) / sigma^2))
    return c0 .+ [f(x, y, z) for x in 1:nx, y in 1:ny, z in 1:nz]
end

function setup_constant_vel_1D_CPML(nt, dt, nx, dx, c0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    vel = VpAcousticCDMaterialProperties(c0 .* ones(nx))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx,), (dx,),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    t0 = 2 / f0
    times = collect(range(0.0; step=dt, length=nt))
    possrcs = zeros(1, 1)
    srctf = zeros(nt, 1)
    srctf[:, 1] .= rickerstf.(times, t0, f0)
    possrcs[1, :] = [lx / 2]
    # receivers
    posrecs = zeros(1, 1)
    posrecs[1, :] = [lx / 3]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, vel
end

function setup_constant_vel_rho_1D_CPML(nt, dt, nx, dx, c0, ρ0, t0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    matprop = VpRhoAcousticVDMaterialProperties(c0 .* ones(nx), ρ0 .* ones(nx))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx,), (dx,),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    times = collect(range(0.0; step=dt, length=nt))
    possrcs = zeros(1, 1)
    srctf = zeros(nt, 1)
    srctf[:, 1] .= gaussderivstf.(times, t0, f0)
    refsrctf = zeros(nt, 1)
    possrcs[1, :] = [lx / 2]
    # receivers
    posrecs = zeros(1, 1)
    posrecs[1, :] = [lx / 3]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, matprop, refsrctf
end

function setup_constant_vel_2D_CPML(nt, dt, nx, ny, dx, dy, c0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    vel = VpAcousticCDMaterialProperties(c0 .* ones(nx, ny))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx, ny), (dx, dy),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    t0 = 2 / f0
    times = collect(range(0.0; step=dt, length=nt))
    possrcs = zeros(1, 2)
    possrcs[1, :] = [lx / 2, ly / 2]
    srctf = zeros(nt, 1)
    srctf[:, 1] .= rickerstf.(times, t0, f0)
    # receivers
    posrecs = zeros(1, 2)
    posrecs[1, :] = [lx / 3, ly / 2]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, vel
end

function setup_constant_vel_rho_2D_CPML(nt, dt, nx, ny, dx, dy, c0, ρ0, t0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    matprop = VpRhoAcousticVDMaterialProperties(c0 .* ones(nx, ny), ρ0 .* ones(nx, ny))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx, ny), (dx, dy),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    times = collect(range(0.0; step=dt, length=nt))
    possrcs = zeros(1, 2)
    possrcs[1, :] = [lx / 2, ly / 2]
    srctf = zeros(nt, 1)
    srctf[:, 1] .= gaussderivstf.(times, t0, f0)
    # receivers
    posrecs = zeros(1, 2)
    posrecs[1, :] = [lx / 3, ly / 2]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, matprop
end

function setup_constant_vel_3D_CPML(nt, dt, nx, ny, nz, dx, dy, dz, c0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    lz = (nz - 1) * dz
    vel = VpAcousticCDMaterialProperties(c0 .* ones(nx, ny, nz))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx, ny, nz), (dx, dy, dz),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    t0 = 2 / f0
    times = collect(range(0.0; step=dt, length=nt))
    possrcs = zeros(1, 3)
    possrcs[1, :] = [lx / 2, ly / 2, lz / 2]
    srctf = zeros(nt, 1)
    srctf[:, 1] .= rickerstf.(times, t0, f0)
    # receivers
    posrecs = zeros(1, 3)
    posrecs[1, :] = [lx / 3, ly / 2, lz / 2]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, vel
end

function analytical_solution_constant_vel_1D(c0, dt, nt, srcs, recs)
    # analytical solution
    times = collect(range(dt; step=dt, length=nt))
    dist = norm(srcs.positions[1, :] .- recs.positions[1, :])
    src = (c0^2) .* srcs.tf[:, 1]
    # Calculate Green's function
    G = times .* 0.0
    for it in 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (2 * c0)
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    return times, Gc
end

function analytical_solution_constant_vel_constant_density_1D(c0, rho, dt, nt, t0, f0, srcs, recs)
    # analytical solution
    times = collect(range(dt / 2; step=dt, length=nt))
    dist = norm(srcs.positions[1, :] .- recs.positions[1, :])
    src = (c0^2 * rho) .* rickerstf.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.0
    for it in 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (2 * c0)
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    return times, Gc
end

function analytical_solution_constant_vel_2D(c0, dt, nt, srcs, recs)
    # analytical solution
    times = collect(range(dt; step=dt, length=nt))
    dist = norm(srcs.positions[1, :] .- recs.positions[1, :])
    src = (c0^2) .* srcs.tf[:, 1]
    # Calculate Green's function
    G = times .* 0.0
    for it in 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    return times, Gc
end

function analytical_solution_constant_vel_constant_density_2D(c0, rho, dt, nt, t0, f0, srcs, recs)
    # analytical solution
    times = collect(range(dt / 2; step=dt, length=nt))
    dist = norm(srcs.positions[1, :] .- recs.positions[1, :])
    src = (c0^2 * rho) .* rickerstf.(times, t0, f0)
    # Calculate Green's function
    G = times .* 0.0
    for it in 1:nt
        # Heaviside function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (2π * c0^2 * sqrt((times[it]^2) - (dist^2 / (c0^2))))
        end
    end
    # Convolve with source term
    Gc = conv(G, src .* dt)
    Gc = Gc[1:nt]

    return times, Gc
end

function analytical_solution_constant_vel_3D(c0, dt, nt, srcs, recs)
    # analytical solution
    times = collect(range(dt; step=dt, length=nt))
    dist = norm(srcs.positions[1, :] .- recs.positions[1, :])
    src = (c0^2) .* srcs.tf[:, 1]
    # Calculate Green's function
    G = times .* 0.0
    for it in 1:nt
        # Delta function
        if (times[it] - dist / c0) >= 0
            G[it] = 1.0 / (4π * c0^2 * dist)
            break
        end
    end
    # Convolve with source term
    Gc = conv(G, src)
    Gc = Gc[1:nt]

    return times, Gc
end

function setup_constant_vel_1D_CPML_Float32(nt, dt, nx, dx, c0, f0, halo, rcoef)
    # constant velocity setup
    lx = (nx - 1) * dx
    vel = VpAcousticCDMaterialProperties(c0 .* ones(Float32, nx))
    # input parameters
    params = InputParametersAcoustic(nt, dt, (nx,), (dx,),
        CPMLBoundaryConditionParameters(; halo=halo, rcoef=rcoef, freeboundtop=false))
    # sources
    t0 = 2f0 / f0
    times = convert.(Float32, collect(range(0.0; step=dt, length=nt)))
    possrcs = zeros(Float32, 1, 1)
    srctf = zeros(Float32, nt, 1)
    srctf[:, 1] .= rickerstf.(times, t0, f0)
    possrcs[1, :] = [lx / 2f0]
    # receivers
    posrecs = zeros(Float32, 1, 1)
    posrecs[1, :] = [lx / 3f0]
    srcs = ScalarSources(possrcs, srctf, f0)
    recs = ScalarReceivers(posrecs, nt; observed=copy(srctf), invcov=Diagonal(ones(Float32, nt)))
    shots = [ScalarShot(; srcs=srcs, recs=recs)]
    return params, shots, vel
end