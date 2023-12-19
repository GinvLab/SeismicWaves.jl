
Base.@kwdef struct ElasticIsoMaterialProperty{N} <: MaterialProperties{N}
    λ::Array{<:AbstractFloat, N}
    μ::Array{<:AbstractFloat, N}
    ρ::Array{<:AbstractFloat, N}
end


# Material properties for 2D simulations
Base.@kwdef struct ElasticIsoMaterialProperty2D <: ElasticIsoMaterialProperty{2}
    λ::Array{<:AbstractFloat,2}
    μ::Array{<:AbstractFloat,2}
    ρ::Array{<:AbstractFloat,2}
    λ_ihalf::Array{<:AbstractFloat,2}
    μ_ihalf::Array{<:AbstractFloat,2}
    μ_jhalf::Array{<:AbstractFloat,2}
    ρ_ihalf_jhalf::Array{<:AbstractFloat,2}
    
    # function ElasticIsoMaterialProperty{N}(ρ::Array{<:AbstractFloat, N},
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



function precomp_elaprop!(matprop::ElasticIsoMaterialProperty{2}; harmonicaver_μ=true)
    
# function precomp_elaprop2D!(ρ,μ,λ,ρ_ihalf_jhalf,μ_ihalf,μ_jhalf,λ_ihalf ;
#                           harmonicaver_μ=true)

    #-------------------------------------------------------------
    # pre-interpolate properties at half distances between nodes
    #-------------------------------------------------------------
    # ρ_ihalf_jhalf (nx-1,nz-1) ??
    # arithmetic mean for ρ
    @. ρ_ihalf_jhalf = (ρ[2:end,2:end]+ρ[2:end,1:end-1]+
        ρ[1:end-1,2:end]+ρ[1:end-1,1:end-1])/4.0
    # μ_ihalf (nx-1,nz) ??
    # μ_ihalf (nx,nz-1) ??
    if harmonicaver_μ==true 
        # harmonic mean for μ
        @. μ_ihalf = 1.0 / ( 1.0/μ[2:end,:] + 1.0 / μ[1:end-1,:] )
        @. μ_jhalf = 1.0 / ( 1.0/μ[:,2:end] + 1.0 / μ[:,1:end-1] )
    else
        # arithmetic mean for μ
        @. μ_ihalf = (μ[2:end,:] + μ[1:end-1,:]) / 2.0
        @. μ_jhalf = (μ[:,2:end] + μ[:,1:end-1]) / 2.0 
    end
    # λ_ihalf (nx-1,nz) ??
    # arithmetic mean for λ
    @. λ_ihalf = (λ[2:end,:] + λ[1:end-1,:]) / 2.0

    return
end
