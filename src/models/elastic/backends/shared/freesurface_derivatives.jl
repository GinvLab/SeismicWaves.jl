# Regular 4th order derivatives for inner points
function ∂x4th_inner(f1, f2, f3, f4, _Δx)
    return (1/24 * f1 - 27/24 * f2 + 27/24 * f3 - 1/24 * f4) * _Δx
end
function ∂y4th_inner(f1, f2, f3, f4, _Δz)
    return (1/24 * f1 - 27/24 * f2 + 27/24 * f3 - 1/24 * f4) * _Δz
end

# Adjusted 4th order derivatives for free surface boundary condition

# Formula #1 of H-AFDA formulation from Kristek et al. (2002)
# `f0` is the value of the function at the free surface, which must be half a grid point above/below the (i,j) point
function formula1_x_left(f, i, j, _Δx, f0)
    return (-352/105 * f0 + 35/8 * f[i, j] - 35/24 * f[i+1, j] + 21/40 * f[i+2, j] - 5/56 * f[i+3, j]) * _Δx
end
function formula1_x_right(f, i, j, _Δx, f0)
    return ( 352/105 * f0 - 35/8 * f[i, j] + 35/24 * f[i-1, j] - 21/40 * f[i-2, j] + 5/56 * f[i-3, j]) * _Δx
end
function formula1_y_top(f, i, j, _Δz, f0)
    return (-352/105 * f0 + 35/8 * f[i, j] - 35/24 * f[i, j+1] + 21/40 * f[i, j+2] - 5/56 * f[i, j+3]) * _Δz
end
function formula1_y_bottom(f, i, j, _Δz, f0)
    return ( 352/105 * f0 - 35/8 * f[i, j] + 35/24 * f[i, j-1] - 21/40 * f[i, j-2] + 5/56 * f[i, j-3]) * _Δz
end

# Formula #2 of H-AFDA formulation from Kristek et al. (2002)
function formula2_x_left(f, i, j, _Δx)
    return (-11/12 * f[i, j] + 17/24 * f[i+1, j] + 3/8 * f[i+2, j] - 5/24 * f[i+3, j] + 1/24 * f[i+4, j]) * _Δx
end
function formula2_x_right(f, i, j, _Δx)
    return ( 11/12 * f[i, j] - 17/24 * f[i-1, j] - 3/8 * f[i-2, j] + 5/24 * f[i-3, j] - 1/24 * f[i-4, j]) * _Δx
end
function formula2_y_top(f, i, j, _Δz)
    return (-11/12 * f[i, j] + 17/24 * f[i, j+1] + 3/8 * f[i, j+2] - 5/24 * f[i, j+3] + 1/24 * f[i, j+4]) * _Δz
end
function formula2_y_bottom(f, i, j, _Δz)
    return ( 11/12 * f[i, j] - 17/24 * f[i, j-1] - 3/8 * f[i, j-2] + 5/24 * f[i, j-3] - 1/24 * f[i, j-4]) * _Δz
end

# Formula #3 of H-AFDA formulation from Kristek et al. (2002)
# `∂f0` is the first derivative of the function at the free surface, which must be 3 half points above/below the (i,j) point
function formula3_x_left(f, i, j, _Δx, ∂f0)
    return (-1/22 * ∂f0) + (-577/528 * f[i-1, j] + 201/176 * f[i, j] - 9/176 * f[i+1, j] + 1/528 * f[i+2, j]) * _Δx
end
function formula3_x_right(f, i, j, _Δx, ∂f0)
    return ( 1/22 * ∂f0) + ( 577/528 * f[i+1, j] - 201/176 * f[i, j] + 9/176 * f[i-1, j] - 1/528 * f[i-2, j]) * _Δx
end
function formula3_y_top(f, i, j, _Δz, ∂f0)
    return (-1/22 * ∂f0) + (-577/528 * f[i, j-1] + 201/176 * f[i, j] - 9/176 * f[i, j+1] + 1/528 * f[i, j+2]) * _Δz
end
function formula3_y_bottom(f, i, j, _Δz, ∂f0)
    return ( 1/22 * ∂f0) + ( 577/528 * f[i, j+1] - 201/176 * f[i, j] + 9/176 * f[i, j-1] - 1/528 * f[i, j-2]) * _Δz
end

# Formula #4 of H-AFDA formulation from Kristek et al. (2002)
# `f0` is the value of the function at the free surface, which must be 3 half points above/below the (i,j) point
function formula4_x_left(f, i, j, _Δx, f0)
    return (16/105 * f0 - 31/24 * f[i-1, j] + 29/24 * f[i, j] - 3/40 * f[i+1, j] + 1/168 * f[i+2, j]) * _Δx
end
function formula4_x_right(f, i, j, _Δx, f0)
    return (-16/105 * f0 + 31/24 * f[i+1, j] - 29/24 * f[i, j] + 3/40 * f[i-1, j] - 1/168 * f[i-2, j]) * _Δx
end
function formula4_y_top(f, i, j, _Δz, f0)
    return (16/105 * f0 - 31/24 * f[i, j-1] + 29/24 * f[i, j] - 3/40 * f[i, j+1] + 1/168 * f[i, j+2]) * _Δz
end
function formula4_y_bottom(f, i, j, _Δz, f0)
    return (-16/105 * f0 + 31/24 * f[i, j+1] - 29/24 * f[i, j] + 3/40 * f[i, j-1] - 1/168 * f[i, j-2]) * _Δz
end

# Wrapper functions to select the appropriate formula based on the field of which we are computing the derivative

function ∂σxx∂x_4th(σxx, i, j, _Δx, nx)
    if i == 1
        # Set missing left point to 0
        return ∂x4th_inner(0, σxx[i, j], σxx[i+1, j], σxx[i+2, j], _Δx)
    elseif i == nx-1
        # Set missing right point to 0
        return ∂x4th_inner(σxx[i-1, j], σxx[i, j], σxx[i+1, j], 0, _Δx)
    else
        # Inner point
        return ∂x4th_inner(σxx[i-1, j], σxx[i, j], σxx[i+1, j], σxx[i+2, j], _Δx)
    end
end

function ∂σzz∂z_4th(σzz, i, j, _Δz, nz)
    if j == 1
        # Free surface at top boundary, half a grid point above
        return formula2_y_top(σzz, i, j, _Δz)
    elseif j == nz-1
        # Set missing bottom point to 0
        return ∂y4th_inner(σzz[i, j-1], σzz[i, j], σzz[i, j+1], 0, _Δz)
    else
        # Inner point
        return ∂y4th_inner(σzz[i, j-1], σzz[i, j], σzz[i, j+1], σzz[i, j+2], _Δz)
    end
end

function ∂σxz∂x_4th(σxz, i, j, _Δz, nx)
    if i == 1
        # Set missing left points to 0
        return ∂x4th_inner(0, 0, σxz[i, j], σxz[i+1, j], _Δz)
    elseif i == 2
        # Set missing left point to 0
        return ∂x4th_inner(0, σxz[i-1, j], σxz[i, j], σxz[i+1, j], _Δz)
    elseif i == nx-1
        # Set missing right point to 0
        return ∂x4th_inner(σxz[i-2, j], σxz[i-1, j], σxz[i, j], 0, _Δz)
    elseif i == nx
        # Set missing right points to 0
        return ∂x4th_inner(σxz[i-2, j], σxz[i-1, j], 0, 0, _Δz)
    else
        # Inner point
        # i-1 because σxz is defined on staggered grid
        return ∂x4th_inner(σxz[i-2, j], σxz[i-1, j], σxz[i, j], σxz[i+1, j], _Δz)
    end
end

function ∂σxz∂z_4th(σxz, i, j, _Δz, nz)
    if j == 1
        # Free surface at top boundary, half a grid point above
        f0 = 0 # σxz = 0 at free surface
        return formula1_y_top(σxz, i, j, _Δz, f0)
    elseif j == 2
        # Free surface at top boundary, 3 half grid points above
        f0 = 0 # σxz = 0 at free surface
        return formula4_y_top(σxz, i, j, _Δz, f0)
    elseif j == nz-1
        # Set missing bottom point to 0
        return ∂y4th_inner(σxz[i, j-2], σxz[i, j-1], σxz[i, j], 0, _Δz)
    elseif j == nz
        # Set missing bottom points to 0
        return ∂y4th_inner(σxz[i, j-2], σxz[i, j-1], 0, 0, _Δz)
    else
        # Inner point
        # j-1 because σxz is defined on staggered grid
        return ∂y4th_inner(σxz[i, j-2], σxz[i, j-1], σxz[i, j], σxz[i, j+1], _Δz)
    end
end

function ∂ux∂x_4th(ux, uz, λ, μ, i, j, _Δx, _Δz, nx, nz)
    if i == 1
        # Set missing left points to 0
        return ∂x4th_inner(0, 0, ux[i, j], ux[i+1, j], _Δx)
    elseif i == 2
        # Set missing left point to 0
        return ∂x4th_inner(0, ux[i-1, j], ux[i, j], ux[i+1, j], _Δx)
    elseif i == nx-1
        # Set missing right point to 0
        return ∂x4th_inner(ux[i-2, j], ux[i-1, j], ux[i, j], 0, _Δx)
    elseif i == nx
        # Set missing right points to 0
        return ∂x4th_inner(ux[i-2, j], ux[i-1, j], 0, 0, _Δx)
    else
        # Inner point
        # i-1 because ux is defined on staggered grid
        return ∂x4th_inner(ux[i-2, j], ux[i-1, j], ux[i, j], ux[i+1, j], _Δx)
    end
end

function ∂uz∂z_4th(ux, uz, λ, μ, i, j, _Δx, _Δz, nx, nz)
    if j == 1
        # On the free surface at top boundary, use ∂ux∂x to compute ∂uz∂z via Hooke's law and free surface condition
        ∂ux∂x = ∂ux∂x_4th(ux, uz, λ, μ, i, j, _Δx, _Δz, nx, nz)
        # σzz = (λ + 2μ) * ∂uz∂z + λ * ∂ux∂x = 0  =>  ∂uz∂z = - (λ / (λ + 2μ)) * ∂ux∂x
        return (-λ[i,j] / (λ[i,j] + 2*μ[i,j])) * ∂ux∂x
    elseif j == 2
        # A grid point inside the domain, but adjacent to the free surface at top boundary
        # Compute ∂uz∂z a grid point before and use it in formula #3
        ∂uz0 = ∂uz∂z_4th(ux, uz, λ, μ, i, j-1, _Δx, _Δz, nx, nz)
        return formula3_y_top(uz, i, j, _Δz, ∂uz0)
    elseif j == nz-1
        # Set missing bottom point to 0
        return ∂y4th_inner(uz[i, j-2], uz[i, j-1], uz[i, j], 0, _Δz)
    elseif j == nz
        # Set missing bottom points to 0
        return ∂y4th_inner(uz[i, j-2], uz[i, j-1], 0, 0, _Δz)
    else
        # Inner point
        # j-1 because uz is defined on staggered grid
        return ∂y4th_inner(uz[i, j-2], uz[i, j-1], uz[i, j], uz[i, j+1], _Δz)
    end
end

function ∂ux∂z_4th(ux, i, j, _Δz, nz)
    if j == 1
        # Free surface at top boundary, half a grid point above
        return formula2_y_top(ux, i, j, _Δz)
    elseif j == nz-1
        # Set missing bottom point to 0
        return ∂y4th_inner(ux[i, j-1], ux[i, j], ux[i, j+1], 0, _Δz)
    else
        # Inner point
        return ∂y4th_inner(ux[i, j-1], ux[i, j], ux[i, j+1], ux[i, j+2], _Δz)
    end
end

function ∂uz∂x_4th(uz, i, j, _Δx, nx)
    if i == 1
        # Set missing left point to 0
        return ∂x4th_inner(0, uz[i, j], uz[i+1, j], uz[i+2, j], _Δx)
    elseif i == nx-1
        # Set missing right point to 0
        return ∂x4th_inner(uz[i-1, j], uz[i, j], uz[i+1, j], 0, _Δx)
    else
        # Inner point
        return ∂x4th_inner(uz[i-1, j], uz[i, j], uz[i+1, j], uz[i+2, j], _Δx)
    end
end