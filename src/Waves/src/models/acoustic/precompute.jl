"""
    @views precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel)

Precomputes factors for isotropic acoustic models.
"""
@views function precompute_fact!(::IsotropicAcousticWaveEquation, model::WaveModel)
    model.fact .= (model.dt^2) .* (model.vel .^ 2)
end

@views function extract_shot(::IsotropicAcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                      # number of sources
    dt2srctf = (model.dt^2 / model.dx) .* srcs.tf        # scaled source time functions (prescaling with boxcar function 1/dx)
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.dx + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.dx + 1], RoundNearestTiesUp)
    end
    
    return possrcs, posrecs, dt2srctf, traces
end

@views function extract_shot(::IsotropicAcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                                 # number of sources
    dt2srctf = (model.dt^2 / model.dx / model.dy) .* srcs.tf        # scaled source time functions (prescaling with boxcar function 1/(dx*dy))
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.dx + 1, srcs.positions[s,2] / model.dy + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.dx + 1, recs.positions[r,2] / model.dy + 1], RoundNearestTiesUp)
    end
    
    return possrcs, posrecs, dt2srctf, traces
end

@views function extract_shot(::IsotropicAcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                                 # number of sources
    dt2srctf = (model.dt^2 / model.dx / model.dy / model.dz) .* srcs.tf        # scaled source time functions (prescaling with boxcar function 1/(dx*dy*dz))
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.dx + 1, srcs.positions[s,2] / model.dy + 1, srcs.positions[s,3] / model.dz + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.dx + 1, recs.positions[r,2] / model.dy + 1, srcs.positions[s,3] / model.dz + 1], RoundNearestTiesUp)
    end
    
    return possrcs, posrecs, dt2srctf, traces
end