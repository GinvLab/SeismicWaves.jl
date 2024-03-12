
struct CPMLCoefficientsAxis
    a::Any
    a_h::Any
    b::Any
    b_h::Any

    function CPMLCoefficientsAxis(halo::Integer, backend::Module,
        sizehalfgrdplusone::Bool=false)
        if sizehalfgrdplusone
            return new(
                backend.zeros(2 * halo),
                backend.zeros(2 * (halo + 1) + 1),
                backend.zeros(2 * halo),
                backend.zeros(2 * (halo + 1) + 1)
            )
        else
            return new(
                backend.zeros(2 * halo),
                backend.zeros(2 * (halo + 1)),
                backend.zeros(2 * halo),
                backend.zeros(2 * (halo + 1))
            )
        end
    end
end

function compute_CPML_coefficientsAxis!(
    cpmlcoeffs::CPMLCoefficientsAxis,
    vel_max::Real,
    dt::Real,
    halo::Integer,
    rcoef::Real,
    thickness::Real,
    f0::Real
)
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    alpha_max = π * f0  # CPML α multiplicative factor (half of dominating angular frequency)
    npower = 2.0  # CPML power coefficient
    d0 = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness)  # damping profile
    a_l, a_r, b_l, b_r = calc_Kab_CPML_staggeredgrid(halo, dt, npower, d0, alpha_max, :startongrd)
    a_hl, a_hr, b_hl, b_hr = calc_Kab_CPML_staggeredgrid(halo, dt, npower, d0, alpha_max, :starthalfgrd)

    copyto!(cpmlcoeffs.a, vcat(a_l, a_r))
    copyto!(cpmlcoeffs.a_h, vcat(a_hl, a_hr))
    copyto!(cpmlcoeffs.b, vcat(b_l, b_r))
    copyto!(cpmlcoeffs.b_h, vcat(b_hl, b_hr))
end

#################################################################

struct CPMLCoefficients
    a_l::Any
    a_r::Any
    a_hl::Any
    a_hr::Any
    b_l::Any
    b_r::Any
    b_hl::Any
    b_hr::Any

    function CPMLCoefficients(halo::Integer, backend::Module,
        sizehalfgrdplusone::Bool=false)
        if sizehalfgrdplusone
            return new(
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo + 1),
                backend.zeros(halo + 1),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo + 1),
                backend.zeros(halo + 1)
            )
        else
            return new(
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo),
                backend.zeros(halo)
            )
        end
    end
end

function calc_Kab_CPML_staggeredgrid(
    halo::Integer,
    dt::Float64,
    npower::Float64,
    d0::Float64,
    alpha_max_pml::Float64,
    onwhere::Symbol;
    K_max_pml::Union{Float64, Nothing}=nothing
)::Tuple{Array{<:Real}, Array{<:Real}, Array{<:Real}, Array{<:Real}}
    @assert halo >= 0.0

    Kab_size = halo
    # shift for half grid coefficients
    if onwhere == :starthalfgrd
        shift_left = 0.0
        shift_right = 0.5
    elseif onwhere == :startongrd
        shift_left = 0.5
        shift_right = 0.0
    else
        error("Wrong onwhere parameter!")
    end

    # distance from edge node
    dist_left = collect(LinRange(shift_left, Kab_size + shift_left - 1, Kab_size))
    dist_right = collect(LinRange(shift_right, Kab_size + shift_right - 1, Kab_size))

    if halo != 0
        normdist_left = reverse(dist_left) ./ (halo - 0.5)
        normdist_right = dist_right ./ (halo - 0.5)
    else
        normdist_left = reverse(dist_left)
        normdist_right = dist_right
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
        return a_left, a_right, b_left, b_right
    else
        return a_left, a_right, b_left, b_right, K_left, K_right
    end
end

#####################################

function compute_CPML_coefficients!(
    cpmlcoeffs::CPMLCoefficients,
    vel_max::Real,
    dt::Real,
    halo::Integer,
    rcoef::Real,
    thickness::Real,
    f0::Real
)
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    alpha_max = π * f0          # CPML α multiplicative factor (half of dominating angular frequency)
    npower = 2.0                # CPML power coefficient
    d0 = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness)     # damping profile
    if halo == 0 # fix for thickness == 0 generating NaNs
        d0 = 0.0
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
    halo::Integer,
    dt::Float64,
    npower::Float64,
    d0::Float64,
    alpha_max_pml::Float64,
    onwhere::String;
    K_max_pml::Union{Float64, Nothing}=nothing
)::Tuple{Array{<:Real}, Array{<:Real}, Array{<:Real}, Array{<:Real}}
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
        return a_left, a_right, b_left, b_right
    else
        return a_left, a_right, b_left, b_right, K_left, K_right
    end
end

#####################################
