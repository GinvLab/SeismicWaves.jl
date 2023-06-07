
Base.@kwdef struct ElasticIsoMaterialProperty{N} <: MaterialProperties{N}
    λ::Array{<:Float64, N}
    μ::Array{<:Float64, N}
    ρ::Array{<:Float64, N}
end



Base.@kwdef struct ElasticIsoMaterialProperty_Compute2D{N} <: MaterialProperties{N}
    λ::Array{<:Float64, N}
    μ::Array{<:Float64, N}
    ρ::Array{<:Float64, N}
    λ_ihalf::Array{<:Float64, N}
    μ_ihalf::Array{<:Float64, N}
    μ_jhalf::Array{<:Float64, N}
    ρ_ihalf_jhalf::Array{<:Float64, N}
end
