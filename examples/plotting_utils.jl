using Plots, Plots.Measures

function plot_nice_heatmap(A; lx=size(A,1), ly=size(A,2), dx=1, dy=1, shift=0)
    heatmap(
        0-shift:dx:lx-shift, 0-shift:dy:ly-shift, A';
        aspect_ratio=:equal, margin=5mm, cmap=:jet1, yflip=true,
        xlims=(0-shift,lx-shift), ylims=(0-shift,ly-shift)
    )
end

function plot_nice_heatmap_grad(A; lx=size(A,1), ly=size(A,2), dx=1, dy=1, shift=0)
    vmax = maximum(abs.(A))
    heatmap(
        0-shift:dx:lx-shift, 0-shift:dy:ly-shift, A';
        aspect_ratio=:equal, margin=5mm, cmap=:RdBu, yflip=true,
        xlims=(0-shift,lx-shift), ylims=(0-shift,ly-shift), clims=(-vmax, vmax)
    )
end

function plot_rays_2D(shots, model, lx, ly, dx, dy, halo; save_file=nothing)
    # plot velocity model
    p = plot_nice_heatmap(model; lx=lx, ly=ly, dx=dx, dy=dy)
    # plot shots
    cmp = palette(:default, length(shots))
    for (i, s) in enumerate(shots)
        nsrcs = size(s.srcs.positions, 1)
        nrecs = size(s.recs.positions, 1)
        # plot rays
        for isrc in 1:nsrcs
            for irec in 1:nrecs
                plot!([s.srcs.positions[isrc, 1], s.recs.positions[irec, 1]], [s.srcs.positions[isrc, 2], s.recs.positions[irec, 2]];
                      color=cmp[i])
            end
        end
        # display sources and receivers positions
        scatter!(s.srcs.positions[:,1], s.srcs.positions[:,2]; markershape=:star4, markersize=5, markercolor=:white)
        scatter!(s.recs.positions[:,1], s.recs.positions[:,2]; markershape=:utriangle, markersize=4, markercolor=:white)
    end
    # display CPML boundaries
    plot!([halo*dx, lx-halo*dx], [halo*dy, halo*dy]; color=:white)
    plot!([halo*dx, lx-halo*dx], [ly-halo*dy, ly-halo*dy]; color=:white)
    plot!([halo*dx, halo*dx], [halo*dy, ly-halo*dy]; color=:white)
    plot!([lx-halo*dx, lx-halo*dx], [halo*dy, ly-halo*dy]; color=:white)
    # set other plotting parameters
    plot!(; legend=nothing)

    if save_file !== nothing
        savefig(save_file)
    else
        display(p)
    end
end