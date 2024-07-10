using Serialization

include("../../models.jl")
include("../../geometries.jl")
include("../../plotting_utils.jl")

# Numerical parameters
nt = 1000
r = 35
dh = dx = dy = 5.0
dt = dh / sqrt(2)
halo = 20
rcoef = 0.0001
nx = 201
ny = 201
lx = (nx - 1) * dx
ly = (ny - 1) * dy

corner1 = 21
corner2 = round(Int, 300 ÷ dx)
l = @layout([A B C])

# Get gradient
gradients = deserialize("grad.dat")
adjgrad_vp = gradients[:, :, 1]
adjgrad_rho = gradients[:, :, 2]
fd_gradients = deserialize("fdgrad.dat")
fdgrad_vp = fd_gradients[:, :, 1]
fdgrad_rho = fd_gradients[:, :, 2]

@show adjgrad_vp[100, 100]
@show adjgrad_rho[100, 100]
@show abs(adjgrad_vp[100, 100] - fdgrad_vp[100, 100])
@show abs(adjgrad_rho[100, 100] - fdgrad_rho[100, 100])
@show abs((adjgrad_vp[100, 100] - fdgrad_vp[100, 100]) / adjgrad_vp[100, 100])
@show abs((adjgrad_rho[100, 100] - fdgrad_rho[100, 100]) / adjgrad_rho[100, 100])

for i in 1:201
    for j in 1:201
        if isnan(adjgrad_vp[i, j])
            @show i, j
        end
    end
end

# Plot adjoint gradient and zoom in
p_grad = plot_zoom(adjgrad_vp, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    adjgrad_vp[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    adjgrad_vp[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Adjoint gradient w.r.t. model velocities"
)
savefig("adjgrad_vp.png")

# Plot adjoint gradient and zoom in
p_grad = plot_zoom(adjgrad_rho, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    adjgrad_rho[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    adjgrad_rho[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Adjoint gradient w.r.t. model densities"
)
savefig("adjgrad_rho.png")

# Plot FD gradient and zoom in
p_grad = plot_zoom(fdgrad_vp, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    fdgrad_vp[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    fdgrad_vp[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Adjoint gradient w.r.t. model velocities"
)
savefig("fdgrad_vp.png")

# Plot FD gradient and zoom in
p_grad = plot_zoom(fdgrad_rho, corner1, plot_nice_heatmap_grad; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    fdgrad_rho[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap_grad;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap_grad(
    fdgrad_rho[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Adjoint gradient w.r.t. model densities"
)
savefig("fdgrad_rho.png")

grad_diff_vp = fdgrad_vp - adjgrad_vp
# Plot relative difference between adjoint and FD grad and zoom in
rel_diff = log10.(abs.(grad_diff_vp ./ fdgrad_vp) * 100)
p_grad = plot_zoom(rel_diff, corner1, plot_nice_heatmap; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    rel_diff[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap(
    rel_diff[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Log10 of relative error % between adjoint and FD gradient"
)
savefig("rel_grad_vp_err.png")

grad_diff_rho = fdgrad_rho - adjgrad_rho
# Plot relative difference between adjoint and FD grad and zoom in
rel_diff = log10.(abs.(grad_diff_rho ./ fdgrad_rho) * 100)
p_grad = plot_zoom(rel_diff, corner1, plot_nice_heatmap; lx=lx, ly=ly, dx=dx, dy=dy)
p_grad_zoom = plot_zoom(
    rel_diff[corner1:end-corner1, corner1:end-corner1],
    corner2,
    plot_nice_heatmap;
    lx=(nx - corner1 * 2) * dx,
    ly=(ny - corner1 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner1
)
p_grad_zoom2 = plot_nice_heatmap(
    rel_diff[corner2:end-corner2, corner2:end-corner2];
    lx=(nx - corner2 * 2) * dx,
    ly=(ny - corner2 * 2) * dy,
    dx=dx,
    dy=dy,
    shift=-dx * corner2
)
p = plot(
    p_grad,
    p_grad_zoom,
    p_grad_zoom2;
    layout=l,
    legend=nothing,
    size=(1500, 500),
    plot_title="Log10 of relative error % between adjoint and FD gradient"
)
savefig("rel_grad_rho_err.png")