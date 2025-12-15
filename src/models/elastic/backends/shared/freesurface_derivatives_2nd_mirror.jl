# Strass imaging scheme for the free surface, 2nd order accurate

# Regular 2th order derivatives for inner points
function ∂x2nd_inner(f1, f2, _Δx)
    return (-f1 + f2) * _Δx
end
function ∂y2nd_inner(f1, f2, _Δz)
    return (-f1 + f2) * _Δz
end

# Wrapper functions to select the appropriate formula based on the field of which we are computing the derivative

Base.@propagate_inbounds function ∂σxx∂x_2nd(σxx, i, j, _Δx, nx)
    return ∂x2nd_inner(σxx[i, j], σxx[i+1, j], _Δx)
end

Base.@propagate_inbounds function ∂σzz∂z_2nd(σzz, i, j, _Δz, nz, freeboundtop)
    return ∂y2nd_inner(σzz[i, j], σzz[i, j+1], _Δz)
end

Base.@propagate_inbounds function ∂σxz∂x_2nd(σxz, i, j, _Δz, nx)
    if i == 1
        # Set missing left point to 0
        return ∂x2nd_inner(0, σxz[i, j], _Δz)
    elseif i == nx
        # Set missing right point to 0
        return ∂x2nd_inner(σxz[i-1, j], 0, _Δz)
    else
        # Inner point
        # i-1 because σxz is defined on staggered grid
        return ∂x2nd_inner(σxz[i-1, j], σxz[i, j], _Δz)
    end
end

Base.@propagate_inbounds function ∂σxz∂z_2nd(σxz, i, j, _Δz, nz, freeboundtop)
    if j == 1
        if freeboundtop
            # Mirror stress values across free surface
            return ∂y2nd_inner(-σxz[i, j], σxz[i, j], _Δz)
        else
            # Set missing top point to 0
            return ∂y2nd_inner(0, σxz[i, j], _Δz)
        end
    elseif j == nz
        # Set missing bottom point to 0
        return ∂y2nd_inner(σxz[i, j-1], 0, _Δz)
    else
        # Inner point
        # j-1 because σxz is defined on staggered grid
        return ∂y2nd_inner(σxz[i, j-1], σxz[i, j], _Δz)
    end
end

Base.@propagate_inbounds function ∂ux∂x_2nd(ux, i, j, _Δx, nx)
    if i == 1
        # Set missing left point to 0
        return ∂x2nd_inner(0, ux[i, j], _Δx)
    elseif i == nx
        # Set missing right point to 0
        return ∂x2nd_inner(ux[i-1, j], 0, _Δx)
    else
        # Inner point
        # i-1 because ux is defined on staggered grid
        return ∂x2nd_inner(ux[i-1, j], ux[i, j], _Δx)
    end
end

Base.@propagate_inbounds function ∂uz∂z_2nd(ux, uz, λ, μ, i, j, _Δx, _Δz, nx, nz, freeboundtop)
    if j == 1
        if freeboundtop
            # On the free surface at top boundary, use ∂ux∂x to compute ∂uz∂z via Hooke's law and free surface condition
            ∂ux∂x = ∂ux∂x_2nd(ux, i, j, _Δx, nx)
            # σzz = (λ + 2μ) * ∂uz∂z + λ * ∂ux∂x = 0  =>  ∂uz∂z = - (λ / (λ + 2μ)) * ∂ux∂x
            return (-λ[i,j] / (λ[i,j] + 2*μ[i,j])) * ∂ux∂x
        else
            # Set missing top points to 0
            return ∂y2nd_inner(0, uz[i, j], _Δz)
        end
    elseif j == nz
        # Set missing bottom points to 0
        return ∂y2nd_inner(uz[i, j-1], 0, _Δz)
    else
        # Inner point
        # j-1 because uz is defined on staggered grid
        return ∂y2nd_inner(uz[i, j-1], uz[i, j], _Δz)
    end
end

Base.@propagate_inbounds function ∂ux∂z_2nd(ux, i, j, _Δz, nz, freeboundtop)
    return ∂y2nd_inner(ux[i, j], ux[i, j+1], _Δz)
end

Base.@propagate_inbounds function ∂uz∂x_2nd(uz, i, j, _Δx, nx)
    return ∂x2nd_inner(uz[i, j], uz[i+1, j], _Δx)
end