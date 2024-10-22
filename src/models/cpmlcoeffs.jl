
struct CPMLCoefficientsAxis{T, V <: AbstractVector{T}}
    a::V
    a_h::V
    b::V
    b_h::V

    function CPMLCoefficientsAxis{T, V}(halo::Int, backend::Module) where {T, V <: AbstractVector{T}}
        return new{T, V}(
            backend.zeros(T, 2 * halo),
            backend.zeros(T, 2 * (halo + 1)),
            backend.zeros(T, 2 * halo),
            backend.zeros(T, 2 * (halo + 1))
        )
    end
end

function compute_CPML_coefficientsAxis!(
    cpmlcoeffs::CPMLCoefficientsAxis{T, V},
    vel_max::T,
    dt::T,
    halo::Int,
    rcoef::T,
    thickness::T,
    f0::T
) where {T, V <: AbstractVector{T}}
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    alpha_max = convert(T, π * f0)          # CPML α multiplicative factor (half of dominating angular frequency)
    npower = convert(T, 2.0)                # CPML power coefficient
    d0 = convert(T, -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness))     # damping profile
    if halo == 0 # fix for thickness == 0 generating NaNs
        d0 = convert(T, 0.0)
    end
    a_l, a_r, b_l, b_r = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, "ongrd")
    a_hl, a_hr, b_hl, b_hr = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, "halfgrd")

    copyto!(cpmlcoeffs.a, vcat(a_l, a_r))
    copyto!(cpmlcoeffs.a_h, vcat(a_hl, a_hr))
    copyto!(cpmlcoeffs.b, vcat(b_l, b_r))
    copyto!(cpmlcoeffs.b_h, vcat(b_hl, b_hr))
end

#################################################################

struct CPMLCoefficients{T, V <: AbstractVector{T}}
    a_l::V
    a_r::V
    a_hl::V
    a_hr::V
    b_l::V
    b_r::V
    b_hl::V
    b_hr::V

    function CPMLCoefficients{T, V}(halo::Int, backend::Module, sizehalfgrdplusone::Bool=false) where {T, V <: AbstractVector{T}}
        if sizehalfgrdplusone
            return new{T, V}(
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo + 1),
                backend.zeros(T, halo + 1),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo + 1),
                backend.zeros(T, halo + 1)
            )
        else
            return new{T, V}(
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo),
                backend.zeros(T, halo)
            )
        end
    end
end

#####################################

function compute_CPML_coefficients!(
    cpmlcoeffs::CPMLCoefficients{T, V},
    vel_max::T,
    dt::T,
    halo::Int,
    rcoef::T,
    thickness::T,
    f0::T
) where {T, V <: AbstractVector{T}}
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    alpha_max = convert(T, π * f0)          # CPML α multiplicative factor (half of dominating angular frequency)
    npower = convert(T, 2.0)                # CPML power coefficient
    d0 = convert(T, -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness))     # damping profile
    if halo == 0 # fix for thickness == 0 generating NaNs
        d0 = convert(T, 0.0)
    end
    a_l, a_r, b_l, b_r = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, "ongrd")
    a_hl, a_hr, b_hl, b_hr = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, "halfgrd")

    copyto!(cpmlcoeffs.a_l, a_l)
    copyto!(cpmlcoeffs.a_r, a_r)
    copyto!(cpmlcoeffs.a_hl, a_hl)
    copyto!(cpmlcoeffs.a_hr, a_hr)
    copyto!(cpmlcoeffs.b_l, b_l)
    copyto!(cpmlcoeffs.b_r, b_r)
    copyto!(cpmlcoeffs.b_hl, b_hl)
    copyto!(cpmlcoeffs.b_hr, b_hr)
end

#####################################

function calc_Kab_CPML(
    halo::Int,
    dt::T,
    npower::T,
    d0::T,
    alpha_max_pml::T,
    onwhere::String;
    K_max_pml::Union{T, Nothing}=nothing
)::Tuple{Array{T}, Array{T}, Array{T}, Array{T}} where {T}
    @assert halo >= 0.0

    Kab_size = halo
    # shift for half grid coefficients
    if onwhere == "halfgrd"
        Kab_size += 1
        shift = 0.5
    elseif onwhere == "ongrd"
        shift = 0.0
    else
        error("Wrong onwhere parameter!")
    end

    # distance from edge node
    dist = collect(LinRange(0 - shift, Kab_size - shift - 1, Kab_size))
    if onwhere == "halfgrd"
        dist[1] = 0
    end
    if halo != 0
        normdist_left = reverse(dist) ./ halo
        normdist_right = dist ./ halo
    else
        normdist_left = reverse(dist)
        normdist_right = dist
    end

    if K_max_pml === nothing
        K_left = 1.0
    else
        K_left = 1.0 .+ (K_max_pml - 1.0) .* (normdist_left .^ npower)
    end
    d_left = d0 .* (normdist_left .^ npower)
    alpha_left = alpha_max_pml .* (1.0 .- normdist_left)
    b_left = exp.(.-(d_left ./ K_left .+ alpha_left) .* dt)
    a_left = d_left .* (b_left .- 1.0) ./ (K_left .* (d_left .+ K_left .* alpha_left))

    if K_max_pml === nothing
        K_right = 1.0
    else
        K_right = 1.0 .+ (K_max_pml - 1.0) .* (normdist_right .^ npower)
    end
    d_right = d0 .* (normdist_right .^ npower)
    alpha_right = alpha_max_pml .* (1.0 .- normdist_right)
    b_right = exp.(.-(d_right ./ K_right .+ alpha_right) .* dt)
    a_right = d_right .* (b_right .- 1.0) ./ (K_right .* (d_right .+ K_right .* alpha_right))

    if K_max_pml === nothing
        return convert.(T, a_left), convert.(T, a_right), convert.(T, b_left), convert.(T, b_right)
    else
        return convert.(T, a_left), convert.(T, a_right), convert.(T, b_left), convert.(T, b_right), convert.(T, K_left), convert.(T, K_right)
    end
end

#####################################
