
Base.@kwdef struct ElasticIsoMaterialProperties{N} <: AbstrElasticIsoMaterialProperties{N}
    λ::Array{<:AbstractFloat, N}
    μ::Array{<:AbstractFloat, N}
    ρ::Array{<:AbstractFloat, N}
end

# Material properties for 2D simulations
Base.@kwdef struct ElasticIsoMaterialProperties2D <: AbstrElasticIsoMaterialProperties{2}
    λ::Array{<:AbstractFloat, 2}
    μ::Array{<:AbstractFloat, 2}
    ρ::Array{<:AbstractFloat, 2}
    λ_ihalf::Array{<:AbstractFloat, 2}
    μ_ihalf::Array{<:AbstractFloat, 2}
    μ_jhalf::Array{<:AbstractFloat, 2}
    ρ_ihalf_jhalf::Array{<:AbstractFloat, 2}

    # function ElasticIsoMaterialProperties{N}(ρ::Array{<:AbstractFloat, N},
    #                                        μ::Array{<:AbstractFloat, N},
    #                                        λ::Array{<:AbstractFloat, N}) where {N}

    #     @assert ndims(λ)==2
    #     nx,ny = size(λ)
    #     tyrho = eltype(λ)
    #     ρ_ihalf_jhalf = zeros(tyrho,nx-1,ny-1)
    #     μ_ihalf = zeros(tyrho,nx-1,ny)
    #     μ_jhalf = zeros(tyrho,nx,  ny-1)
    #     λ_ihalf = zeros(tyrho,nx-1,ny)
    #     precomp_elaprop2D!(ρ,μ,λ,ρ_ihalf_jhalf,μ_ihalf,μ_jhalf,λ_ihalf ;
    #                        harmonicaver_μ=true)

    #     return new{N}(λ,μ,ρ,λ_ihalf,μ_ihalf,μ_jhalf,ρ_ihalf_jhalf)
    # end
end

function precomp_elaprop!(matprop::ElasticIsoMaterialProperties2D; harmonicaver_μ=true)

    # function precomp_elaprop2D!(ρ,μ,λ,ρ_ihalf_jhalf,μ_ihalf,μ_jhalf,λ_ihalf ;
    #                           harmonicaver_μ=true)

    #-------------------------------------------------------------
    # pre-interpolate properties at half distances between nodes
    #-------------------------------------------------------------
    # ρ_ihalf_jhalf (nx-1,nz-1) ??
    # arithmetic mean for ρ
    @. matprop.ρ_ihalf_jhalf = (matprop.ρ[2:end, 2:end] + matprop.ρ[2:end, 1:end-1] +
                                matprop.ρ[1:end-1, 2:end] + matprop.ρ[1:end-1, 1:end-1]) / 4.0
    # μ_ihalf (nx-1,nz) ??
    # μ_ihalf (nx,nz-1) ??
    if harmonicaver_μ == true
        # harmonic mean for μ
        @. matprop.μ_ihalf = 1.0 / (1.0 / matprop.μ[2:end, :] + 1.0 / matprop.μ[1:end-1, :])
        @. matprop.μ_jhalf = 1.0 / (1.0 / matprop.μ[:, 2:end] + 1.0 / matprop.μ[:, 1:end-1])
    else
        # arithmetic mean for μ
        @. matprop.μ_ihalf = (matprop.μ[2:end, :] + matprop.μ[1:end-1, :]) / 2.0
        @. matprop.μ_jhalf = (matprop.μ[:, 2:end] + matprop.μ[:, 1:end-1]) / 2.0
    end
    # λ_ihalf (nx-1,nz) ??
    # arithmetic mean for λ
    @. matprop.λ_ihalf = (matprop.λ[2:end, :] + matprop.λ[1:end-1, :]) / 2.0

    return
end
