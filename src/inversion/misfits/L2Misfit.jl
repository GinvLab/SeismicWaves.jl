Base.@kwdef struct L2Misfit <: AbstractMisfit
    regularization::Union{Nothing, AbstractRegularization}
end

function (misfit::L2Misfit)(recs::ScalarReceivers{T}, matprop::MaterialProperties{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = dot(residuals, recs.invcov, residuals) / 2
    # Add regularization if needed
    if misfit.regularization !== nothing
        msf += misfit.regularization(matprop)
    end
    return msf
end

function (misfit::L2Misfit)(recs::VectorReceivers{T, N}, matprop::MaterialProperties{T}) where {T, N}
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = sum([dot(residuals[:,i,:], recs.invcov, residuals[:,i,:]) for i in 1:N]) / 2
    # Add regularization if needed
    if misfit.regularization !== nothing
        msf += misfit.regularization(matprop)
    end
    return msf
end

function dχ_du(_::L2Misfit, recs::ScalarReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    return recs.invcov * residuals
end

function dχ_du(_::L2Misfit, recs::VectorReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    for d in axes(residuals, 2)
        residuals[:,d,:] .= recs.invcov * residuals[:,d,:]
    end
    return residuals
end