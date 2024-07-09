@doc """

$(TYPEDEF)

Material properties for elastic isotropic simulation.

$(TYPEDFIELDS)
"""
Base.@kwdef struct ElasticIsoMaterialProperties{T, N} <: AbstrElasticIsoMaterialProperties{T, N}
    "First Lamé parameter"
    λ::Array{T, N}
    "Second Lamé parameter (shear modulus)"
    μ::Array{T, N}
    "Density"
    ρ::Array{T, N}
end
