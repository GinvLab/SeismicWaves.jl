struct VpAcousticCDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
end

struct VpRhoAcousticVDMaterialProperty{N} <: MaterialProperties{N}
    vp::Array{<:Float64, N}
    rho::Array{<:Float64, N}
    rho_stag::NTuple{N, Array{<:Float64, N}}

    function VpRhoAcousticVDMaterialProperty{N}(vp::Array{<:Float64, N}, rho::Array{<:Float64, N}; interp=interp_avg) where {N}
        new(vp, rho, Tuple(interp(rho, i) for i in 1:N))
    end
end
