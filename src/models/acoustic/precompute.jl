"""
    @views precompute_fact!(::AcousticWaveEquation, model::WaveModel)

Precomputes factors for isotropic acoustic models.
"""
@views function precompute_fact!(::AcousticWaveEquation, model::WaveModel)
    model.fact .= (model.dt^2) .* (model.vel .^ 2)
end

@views function prescale_srctf!(dt2srctf, possrcs, fact)
    nsrcs = size(dt2srctf, 2)
    for s = 1:nsrcs
        dt2srctf[:,s] .*= fact[possrcs[s,:]...]
    end
end

@views function extract_shot(::AcousticWaveEquation, model::WaveModel1D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                      # number of sources
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.Δs[1] + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.Δs[1] + 1], RoundNearestTiesUp)
    end
    # prescale with boxcar function 1/dx
    dt2srctf = srcs.tf ./ model.Δs[1]
    # prescale source time function with fact in source positions
    prescale_srctf!(dt2srctf, possrcs, model.fact)
    
    return possrcs, posrecs, dt2srctf, traces
end

@views function extract_shot(::AcousticWaveEquation, model::WaveModel2D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                                 # number of sources
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.Δs[1] + 1, srcs.positions[s,2] / model.Δs[2] + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.Δs[1] + 1, recs.positions[r,2] / model.Δs[2] + 1], RoundNearestTiesUp)
    end
    # prescale with boxcar function 1/(dx*dy)
    dt2srctf = srcs.tf ./ (model.Δs[1] * model.Δs[2])
    # prescale source time function with fact in source positions
    prescale_srctf!(dt2srctf, possrcs, model.fact)
    
    return possrcs, posrecs, dt2srctf, traces
end

@views function extract_shot(::AcousticWaveEquation, model::WaveModel3D, srcs::Sources{<:Real}, recs::Receivers{<:Real})
    # source time functions
    nsrcs = size(srcs.positions, 1)                                 # number of sources
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))           # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / model.Δs[1] + 1, srcs.positions[s,2] / model.Δs[2] + 1, srcs.positions[s,3] / model.Δs[3] + 1], RoundNearestTiesUp)
    end
    # receivers traces
    nrecs = size(recs.positions, 1)                      # number of receivers
    traces = zeros(model.nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / model.Δs[1] + 1, recs.positions[r,2] / model.Δs[2] + 1, recs.positions[r,3] / model.Δs[3] + 1], RoundNearestTiesUp)
    end
    # prescale with boxcar function 1/(dx*dy*dz)
    dt2srctf = srcs.tf ./ (model.Δs[1] * model.Δs[2] * model.Δs[3])
    # prescale source time function with fact in source positions
    prescale_srctf!(dt2srctf, possrcs, model.fact)
    
    return possrcs, posrecs, dt2srctf, traces
end