Base.@kwdef struct ElasticIsoMaterialProperty{N} <: MaterialProperties{N}
    λ::Array{<:Float64, N}
    μ::Array{<:Float64, N}
    ρ::Array{<:Float64, N}
end
