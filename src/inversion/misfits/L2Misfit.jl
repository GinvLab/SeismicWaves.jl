Base.@kwdef struct L2Misfit <: AbstractMisfit
    regularization::Union{Nothing, AbstractRegularization}
end

function (misfit::L2Misfit)(recs::ScalarReceivers, matprop::MaterialProperties)
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2.0
        end
        mask .-= 1.0
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

function dχ_du(_::L2Misfit, recs::ScalarReceivers)
    # Compute residuals
    residuals = recs.seismograms - recs.observed
    # Window residuals using mask
    mask = ones(size(residuals, 1))
    if length(recs.windows) > 0
        for wnd in recs.windows
            mask[wnd.first:wnd.second] .= 2.0
        end
        mask .-= 1.0
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    return recs.invcov * residuals
end