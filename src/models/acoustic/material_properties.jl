struct VpAcousticCDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
end

mutable struct VpRhoAcousticVDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
    rho::Array{<:Float64, N}
    rho_stag::Vector{Array{<:Float64, N}}
    rho_stag_jacob::Vector{Array{<:Float64}}
    interp_method::InterpolationMethod

    function VpRhoAcousticVDMaterialProperty{N}(vp::Array{<:Float64, N}, rho::Array{<:Float64, N}; interp_method::InterpolationMethod=ArithmeticAverageInterpolation(2)) where {N}
        new(vp, rho, interpolate(rho, interp_method), interpolate_jacobian(rho, interp_method), interp_method)
    end
end

function VpRhoAcousticVDMaterialProperty(vp, rho; kwargs...)
    return VpRhoAcousticVDMaterialProperty{length(size(vp))}(vp, rho; kwargs...)
end

interpolate(a::Array{<:Float64, N}, interp_method) where {N} = collect(interp(interp_method, a, i) for i in 1:N)
interpolate_jacobian(a::Array{<:Float64, N}, interp_method) where {N} = collect(jacobian(interp_method, a, i) for i in 1:N)

function interpolate!(matprop::VpRhoAcousticVDMaterialProperty{N}) where {N}
    matprop.rho_stag = interpolate(matprop.rho, matprop.interp_method)
    matprop.rho_stag_jacob = interpolate_jacobian(matprop.rho, matprop.interp_method)
end