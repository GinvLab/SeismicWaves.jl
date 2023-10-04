using Serialization

include("../../models.jl")
include("../../geometries.jl")
include("../../plotting_utils.jl")

# Numerical parameters
nt = 1000
r = 35
dh = dx = dy = 5.0
dt = dh / sqrt(2) / c0max
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
adjgrad_vp = gradients[:,:,1]
adjgrad_rho = gradients[:,:,2]

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