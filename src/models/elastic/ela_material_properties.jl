
Base.@kwdef struct ElasticIsoMaterialProperties{T, N} <: AbstrElasticIsoMaterialProperties{T, N}
    λ::Array{T, N}
    μ::Array{T, N}
    ρ::Array{T, N}
end
