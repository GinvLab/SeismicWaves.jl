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
        msf += regularization(matprop)
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

Base.@kwdef struct ZerothOrderTikhonovRegularization{M <: MaterialProperties} <: AbstractRegularization
    matprop_prior::M
    alpha::Real
end

function (regularization::ZerothOrderTikhonovRegularization{VpAcousticCDMaterialProperty{N}})(matprop::VpAcousticCDMaterialProperty{N}) where {N}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    vp_norm_sq = norm(vp - vp_prior)^2
    return alpha / 2 * vp_norm_sq
end

function (regularization::ZerothOrderTikhonovRegularization{VpRhoAcousticVDMaterialProperty{N}})(matprop::VpRhoAcousticVDMaterialProperty{N}) where {N}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    vp_norm_sq = norm(vp - vp_prior)^2
    rho_norm_sq = norm(rho - rho_prior)^2
    return alpha / 2 * (vp_norm_sq + rho_norm_sq)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{VpAcousticCDMaterialProperty{N}}, matprop::VpAcousticCDMaterialProperty{N}) where {N}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    return alpha * (vp - vp_prior)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{VpRhoAcousticVDMaterialProperty{N}}, matprop::VpRhoAcousticVDMaterialProperty{N}) where {N}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    return alpha * (vp - vp_prior), alpha * (rho - rho_prior)
end