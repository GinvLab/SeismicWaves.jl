@doc """

$(TYPEDEF)

Material properties for elastic isotropic simulation.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct ElasticIsoMaterialProperties{T, N} <: AbstrElasticIsoMaterialProperties{T, N}
    "First Lamé parameter"
    λ::Array{T, N}
    "Second Lamé parameter (shear modulus)"
    μ::Array{T, N}
    "Density"
    ρ::Array{T, N}
    "Interpolation method for density"
    interp_method_ρ::AbstractInterpolationMethod = ArithmeticAverageInterpolation()
    "Interpolation method for density"
    interp_method_λ::AbstractInterpolationMethod = HarmonicAverageInterpolation()
    "Interpolation method for density"
    interp_method_μ::AbstractInterpolationMethod = ArithmeticAverageInterpolation()
end
