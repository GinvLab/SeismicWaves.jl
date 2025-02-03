Base.@kwdef struct ZerothOrderTikhonovRegularization{T, M <: MaterialProperties{T}} <: AbstractRegularization
    matprop_prior::M
    alpha::T
end

function (regularization::ZerothOrderTikhonovRegularization{T, M})(matprop::M) where {T, N, M <: VpAcousticCDMaterialProperties{T, N}}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    vp_norm_sq = norm(vp - vp_prior)^2
    return regularization.alpha / 2 * vp_norm_sq
end

function (regularization::ZerothOrderTikhonovRegularization{T, M})(matprop::M) where {T, N, M <: VpRhoAcousticVDMaterialProperties{T, N}}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    vp_norm_sq = norm(vp - vp_prior)^2
    rho_norm_sq = norm(rho - rho_prior)^2
    return regularization.alpha / 2 * (vp_norm_sq + rho_norm_sq)
end

function (regularization::ZerothOrderTikhonovRegularization{T, M})(matprop::M) where {T, N, M <: ElasticIsoMaterialProperties{T, N}}
    ρ, λ, μ = matprop.ρ, matprop.λ, matprop.μ
    ρ_prior, λ_prior, μ_prior = regularization.matprop_prior.ρ, regularization.matprop_prior.λ, regularization.matprop_prior.μ
    ρ_norm_sq, λ_norm_sq, μ_norm_sq = norm(ρ - ρ_prior)^2, norm(λ - λ_prior)^2, norm(μ - μ_prior)^2
    return regularization.alpha / 2 * (ρ_norm_sq + λ_norm_sq + μ_norm_sq)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{T, M}, matprop::M) where {T, N, M <: VpAcousticCDMaterialProperties{T, N}}
    vp = matprop.vp
    vp_prior = regularization.matprop_prior.vp
    return @. regularization.alpha * (vp - vp_prior)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{T, M}, matprop::M) where {T, N, M <: VpRhoAcousticVDMaterialProperties{T, N}}
    vp = matprop.vp
    rho = matprop.rho
    vp_prior = regularization.matprop_prior.vp
    rho_prior = regularization.matprop_prior.rho
    return @. regularization.alpha * (vp - vp_prior), regularization.alpha * (rho - rho_prior)
end

function dχ_dm(regularization::ZerothOrderTikhonovRegularization{T, M}, matprop::M) where {T, N, M <: ElasticIsoMaterialProperties{T, N}}
    ρ, λ, μ = matprop.ρ, matprop.λ, matprop.μ
    ρ_prior, λ_prior, μ_prior = regularization.matprop_prior.ρ, regularization.matprop_prior.λ, regularization.matprop_prior.μ
    return @. regularization.alpha * (ρ - ρ_prior), regularization.alpha * (λ - λ_prior), regularization.alpha * (μ - μ_prior)
end