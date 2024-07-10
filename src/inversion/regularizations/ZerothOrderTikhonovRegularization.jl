Base.@kwdef struct ZerothOrderTikhonovRegularization{T, M <: MaterialProperties{T}} <: AbstractRegularization
    matprop_prior::M
    alpha::T
end

function (regularization::ZerothOrderTikhonovRegularization{T, VpAcousticCDMaterialProperties{T, N}})(matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    vp_norm_sq = norm(vp - vp_prior)^2
    return regularization.alpha / 2 * vp_norm_sq
end

function (regularization::ZerothOrderTikhonovRegularization{T, VpRhoAcousticVDMaterialProperties{T, N}})(matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    vp_norm_sq = norm(vp - vp_prior)^2
    rho_norm_sq = norm(rho - rho_prior)^2
    return regularization.alpha / 2 * (vp_norm_sq + rho_norm_sq)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{T, VpAcousticCDMaterialProperties{T, N}}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    return regularization.alpha * (vp - vp_prior)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{T, VpRhoAcousticVDMaterialProperties{T, N}}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    return regularization.alpha * (vp - vp_prior), regularization.alpha * (rho - rho_prior)
end