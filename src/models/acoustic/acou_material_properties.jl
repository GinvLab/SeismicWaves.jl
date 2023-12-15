
@doc """

$(TYPEDEF)

Material properties for acoustic constant-density simulation.

$(TYPEDFIELDS)
"""
struct VpAcousticCDMaterialProperty{N} <: MaterialProperties{N}
    "P-wave velocity"
    vp::Array{<:Float64, N}
end


@doc """
$(TYPEDEF)

Material properties for acoustic variable-density simulation.

$(TYPEDFIELDS)
"""
mutable struct VpRhoAcousticVDMaterialProperty{N} <: MaterialProperties{N}
    "P-wave velocity"
    vp::Array{<:Float64, N}
    "Density"
    rho::Array{<:Float64, N}
    "Interpolation method"
    interp_method::InterpolationMethod

    @doc """
        VpRhoAcousticVDMaterialProperty{N}(
          vp::Array{<:Float64, N},
          rho::Array{<:Float64, N};
          interp_method::InterpolationMethod=ArithmeticAverageInterpolation(2)
        ) where {N}

    Constructor for material properties for acoustic variable-density simulation.
    """
    function VpRhoAcousticVDMaterialProperty{N}(
        vp::Array{<:Float64, N},
        rho::Array{<:Float64, N};
        interp_method::InterpolationMethod=ArithmeticAverageInterpolation(2)
    ) where {N}
        return new(vp, rho, interp_method)
    end
end


@doc """
$(SIGNATURES)

Constructor to avoid specifying dimensions to create material properties for acoustic variable-density simulation.
"""
VpRhoAcousticVDMaterialProperty(vp, rho; kwargs...) = VpRhoAcousticVDMaterialProperty{length(size(vp))}(vp, rho; kwargs...)

interpolate(a::Array{<:Float64, N}, interp_method) where {N} = collect(interp(interp_method, a, i) for i in 1:N)
interpolate_jacobian(a::Array{<:Float64, N}, interp_method) where {N} = collect(jacobian(interp_method, a, i) for i in 1:N)
