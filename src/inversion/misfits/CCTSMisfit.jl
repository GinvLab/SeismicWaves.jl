struct CCTSMisfit{T, D, OA <: AbstractArray{T, D}} <: AbstractMisfit{T}
    dt::T
    std::T
    observed::OA

    function CCTSMisfit(dt::T; std::T=one(T), observed::AbstractArray{T, D}=zeros(T, 0, 0)) where {T, D}
        if !(D == 2 || D == 3)
            throw(ArgumentError("Observed data must be a 2D or 3D array!"))
        end
        @assert dt > 0 "Timestep size must be positive!"
        @assert std > 0 "Standard deviation must be positive!"

        return new{T, D, typeof(observed)}(dt, std, observed)
    end
end

function calcmisfit(shotmisfit::CCTSMisfit{T, 2}, recs::ScalarReceivers{T}) where {T}
    # Timestep size
    dt = shotmisfit.dt
    nt = size(recs.seismograms, 1)
    # Compute cross-correlation time series
    misfit = zero(T)
    for r in axes(recs.seismograms, 2)
        dτ = (argmax(abs.(crosscov(shotmisfit.observed[:, r], recs.seismograms[:, r]))) - nt) * dt
        misfit += dτ^2 / 2
    end
    return misfit
end

function calcmisfit(shotmisfit::CCTSMisfit{T, 3}, recs::VectorReceivers{T, N}) where {T, N}
    # Timestep size
    dt = shotmisfit.dt
    nt = size(recs.seismograms, 1)
    # Compute cross-correlation time series
    misfit = zero(T)
    for r in axes(recs.seismograms, 3)
        for d in axes(recs.seismograms, 2)
            dτ = (argmax(abs.(crosscov(shotmisfit.observed[:, d, r], recs.seismograms[:, d, r]))) - nt) * dt
            misfit += dτ^2 / 2
        end
    end
    return misfit
end


function ∂χ_∂u(shotmisfit::CCTSMisfit{T, 2}, recs::ScalarReceivers{T}) where {T}
    # Compute misfit
    dt = shotmisfit.dt
    nt = size(recs.seismograms, 1)
    # Compute adjoint source
    ∂s∂t = zero(recs.seismograms)
    ∂s∂t[2:end-1, :] .= (recs.seismograms[3:end, :] .- recs.seismograms[1:end-2, :]) ./ (2 * dt)  # time derivative of synthetic seismograms
    ∂s∂t[1, :] .= (recs.seismograms[2, :] .- 0) ./ (2 * dt)
    times = collect(range(dt; step=dt, length=nt))
    for r in axes(recs.seismograms, 2)
        dτ = (argmax(abs.(crosscov(shotmisfit.observed[:, r], recs.seismograms[:, r]))) - nt) * dt
        norm_N = integrate(times, ∂s∂t[:, r] .^ 2)
        ∂s∂t[:, r] .= (dτ / norm_N) .* ∂s∂t[:, r]
    end
    return ∂s∂t
end

function ∂χ_∂u(shotmisfit::CCTSMisfit{T, 3}, recs::VectorReceivers{T, N}) where {T, N}
    # Compute misfit
    dt = shotmisfit.dt
    nt = size(recs.seismograms, 1)
    # Compute adjoint source
    ∂s∂t = zero(recs.seismograms)
    ∂s∂t[2:end-1, :, :] .= (recs.seismograms[3:end, :, :] .- recs.seismograms[1:end-2, :, :]) ./ (2 * dt)  # time derivative of synthetic seismograms
    ∂s∂t[1, :, :] .= (recs.seismograms[2, :, :] .- 0) ./ (2 * dt)
    times = collect(range(dt; step=dt, length=nt))
    for r in axes(recs.seismograms, 3)
        for d in axes(recs.seismograms, 2)
            dτ = (argmax(abs.(crosscov(shotmisfit.observed[:, d, r], recs.seismograms[:, d, r]))) - nt) * dt
            norm_N = integrate(times, ∂s∂t[:, d, r] .^ 2)
            ∂s∂t[:, d, r] .= (dτ / norm_N) .* ∂s∂t[:, d, r]
        end
    end
    return ∂s∂t
end
