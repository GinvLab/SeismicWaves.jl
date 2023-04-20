
@views precompute_fact!(model::AcousticCDWaveSimul) = copyto!(model.fact, (model.dt^2) .* (model.matprop.vp .^ 2))

@views function prescale_srctf!(dt2srctf, possrcs, model)
    nsrcs = size(dt2srctf, 2)
    for s in 1:nsrcs
        dt2srctf[:, s] .*= (model.dt^2) .* (model.matprop.vp[possrcs[s, :]...] .^ 2)
    end
end

@views function find_nearest_grid_points(model::WaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                      # number of sources
    ncoos = size(srcs.positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [srcs.positions[s, i] / model.gridspacing[i] + 1 for i in 1:ncoos]
        possrcs[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                      # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    ncoor = size(recs.positions, 2)
    for r in 1:nrecs
        tmp = [recs.positions[r, i] / model.gridspacing[i] + 1 for i in 1:ncoor]
        posrecs[r, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end

    return possrcs, posrecs, traces
end

@views function setup_shot(model::AcousticCDWaveSimul, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    possrcs, posrecs, traces = find_nearest_grid_points(model, srcs, recs)
    # prescale with boxcar function 1/dx, 1/(dx*dy) or 1/(dx*dy*dz)
    dt2srctf = srcs.tf ./ prod(model.gridspacing)
    # prescale source time function with fact in source positions
    prescale_srctf!(dt2srctf, possrcs, model)

    return possrcs, posrecs, dt2srctf, traces
end
