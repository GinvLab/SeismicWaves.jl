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
    interp_method_ρ::InterpolationMethod = ArithmeticAverageInterpolation(2)
    "Interpolation method for density"
    interp_method_λ::InterpolationMethod = ArithmeticAverageInterpolation(2)
    "Interpolation method for density"
    interp_method_μ::InterpolationMethod = ArithmeticAverageInterpolation(2)
end
