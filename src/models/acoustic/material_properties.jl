struct VpAcousticCDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
end

mutable struct VpRhoAcousticVDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
    rho::Array{<:Float64, N}
    interp_method::InterpolationMethod

    function VpRhoAcousticVDMaterialProperty{N}(
        vp::Array{<:Float64, N},
        rho::Array{<:Float64, N};
        interp_method::InterpolationMethod=ArithmeticAverageInterpolation(2)
    ) where {N}
        return new(vp, rho, interp_method)
    end
end

VpRhoAcousticVDMaterialProperty(vp, rho; kwargs...) = VpRhoAcousticVDMaterialProperty{length(size(vp))}(vp, rho; kwargs...)

interpolate(a::Array{<:Float64, N}, interp_method) where {N} = collect(interp(interp_method, a, i) for i in 1:N)
interpolate_jacobian(a::Array{<:Float64, N}, interp_method) where {N} = collect(jacobian(interp_method, a, i) for i in 1:N)
