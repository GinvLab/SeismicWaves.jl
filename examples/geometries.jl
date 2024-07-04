using SeismicWaves

include("plotting_utils.jl")

function linear_2D_geometry(nshots, model, f0, nt, srctf, dd, lx, ly, dx, dy, halo; plot_geometry=false, save_file=nothing)
    # shots definition
    shots = Vector{Shot{Float64}}()  #Pair{ScalarSources, ScalarReceivers}}()
    # straight line positions
    xs = (lx / 2) .- dd .* ((nshots + 1) / 2 .- collect(1:nshots))
    ysrc = (halo + 10) * dy
    yrec = ly - ysrc

    for i in 1:nshots
        # sources definition
        possrcs = reshape([xs[i], ysrc], 1, 2)
        srcs = ScalarSources(
            possrcs,
            reshape(srctf, nt, 1),
            f0
        )

        # receivers definition
        nrecs = nshots
        posrecs = hcat(xs, fill(yrec, nrecs))
        recs = ScalarReceivers(
            posrecs,
            nt
        )

        # add pair as shot
        push!(shots, Shot(; srcs=srcs, recs=recs)) # srcs => recs)
    end

    if plot_geometry
        plot_rays_2D(shots, model, lx, ly, dx, dy, halo; save_file=save_file)
    end

    return shots
end
