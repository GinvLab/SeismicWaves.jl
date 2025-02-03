
@doc """

$(TYPEDEF)

Material properties for acoustic constant-density simulation.

$(TYPEDFIELDS)
"""
struct VpAcousticCDMaterialProperties{T, N} <: MaterialProperties{T, N}
    "P-wave velocity"
    vp::Array{T, N}
end

@doc """
$(TYPEDEF)

Material properties for acoustic variable-density simulation.

$(TYPEDFIELDS)
"""
mutable struct VpRhoAcousticVDMaterialProperties{T, N} <: MaterialProperties{T, N}
    "P-wave velocity"
    vp::Array{T, N}
    "Density"
    rho::Array{T, N}
    "Interpolation method"
    interp_method::AbstractInterpolationMethod

    @doc """
        VpRhoAcousticVDMaterialProperties(
          vp::Array{T, N},
          rho::Array{T, N};
          interp_method::InterpolationMethod=ArithmeticAverageInterpolation()
        ) where {T, N}

    Constructor for material properties for acoustic variable-density simulation.
    """
    function VpRhoAcousticVDMaterialProperties(
        vp::Array{T, N},
        rho::Array{T, N};
        interp_method::AbstractInterpolationMethod=ArithmeticAverageInterpolation()
    ) where {T, N}
        return new{T, N}(vp, rho, interp_method)
    end
end