
Base.@kwdef struct ElasticIsoMaterialProperty{N} <: MaterialProperties{N}
    λ::Array{<:AbstractFloat, N}
    μ::Array{<:AbstractFloat, N}
    ρ::Array{<:AbstractFloat, N}
end



Base.@kwdef struct ElasticIsoMaterialProperty_Compute2D{N} <: MaterialProperties{N}
    λ::Array{<:AbstractFloat, N}
    μ::Array{<:AbstractFloat, N}
    ρ::Array{<:AbstractFloat, N}
    λ_ihalf::Array{<:AbstractFloat, N}
    μ_ihalf::Array{<:AbstractFloat, N}
    μ_jhalf::Array{<:AbstractFloat, N}
    ρ_ihalf_jhalf::Array{<:AbstractFloat, N}
end
