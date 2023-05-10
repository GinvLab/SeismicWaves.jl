function gaussian_vel_1D(nx, c0, c0max, r, origin=(nx+1)/2)
    sigma = r / 3
    amp = c0max - c0
    f(x) = amp * exp(-(0.5 * (x - origin)^2 / sigma^2))
    return c0 .+ [f(x) for x in 1:nx]
end

function gaussian_vel_2D(nx, ny, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2])
    sigma = r / 3
    amp = c0max - c0
    f(x,y) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2) / sigma^2))
    return c0 .+ [f(x,y) for x in 1:nx, y in 1:ny]
end

function gaussian_vel_3D(nx, ny, nz, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2, (nz+1)/2])
    sigma = r / 3
    amp = c0max - c0
    f(x,y,z) = amp * exp(-(0.5 * ((x - origin[1])^2 + (y - origin[2])^2 + (z - origin[3])^2) / sigma^2))
    return c0 .+ [f(x,y,z) for x in 1:nx, y in 1:ny, z in 1:nz]
end

function disc_vel_1D(nx, c0, c0max, r, origin=(nx+1)/2)
    vel = c0 .* ones(nx)
    for i in 1:nx
        if norm(origin - i) <= r
            vel[i] = c0max
        end
    end
    return vel
end

function disc_vel_2D(nx, ny, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2])
    vel = c0 .* ones(nx, ny)
    for i in 1:nx
        for j in 1:ny
            if norm(origin .- [i, j]) <= r
                vel[i,j] = c0max
            end
        end
    end
    return vel
end

function disc_vel_3D(nx, ny, nz, c0, c0max, r, origin=[(nx+1)/2, (ny+1)/2, (nz+1)/2])
    vel = c0 .* ones(nx, ny, nz)
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                if norm(origin .- [i, j, k]) <= r
                    vel[i,j,k] = c0max
                end
            end
        end
    end
    return vel
end