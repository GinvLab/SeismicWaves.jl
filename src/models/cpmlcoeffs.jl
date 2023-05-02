struct CPMLCoefficients
    a_l::Any
    a_r::Any
    a_hl::Any
    a_hr::Any
    b_K_l::Any
    b_K_r::Any
    b_K_hl::Any
    b_K_hr::Any

    function CPMLCoefficients(halo::Integer, backend::Module)
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
    end
end

# Default type constructor
CPMLCoefficients(halo) = CPMLCoefficients{Float64}(halo)

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
    K_max = 1.0                 # CPML K coefficient value
    d0 = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness)     # damping profile
    a_l, a_r, b_K_l, b_K_r = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, K_max, "ongrd")
    a_hl, a_hr, b_K_hl, b_K_hr = calc_Kab_CPML(halo, dt, npower, d0, alpha_max, K_max, "halfgrd")

    copyto!(cpmlcoeffs.a_l, a_l)
    copyto!(cpmlcoeffs.a_r, a_r)
    copyto!(cpmlcoeffs.a_hl, a_hl)
    copyto!(cpmlcoeffs.a_hr, a_hr)
    copyto!(cpmlcoeffs.b_K_l, b_K_l)
    copyto!(cpmlcoeffs.b_K_r, b_K_r)
    copyto!(cpmlcoeffs.b_K_hl, b_K_hl)
    copyto!(cpmlcoeffs.b_K_hr, b_K_hr)
end

function calc_Kab_CPML(
    halo::Integer,
    dt::Float64,
    npower::Float64,
    d0::Float64,
    alpha_max_pml::Float64,
    K_max_pml::Float64,
    onwhere::String
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
    normdist_left = reverse(dist) ./ Kab_size
    normdist_right = dist ./ Kab_size

    d_left = d0 .* (normdist_left .^ npower)
    alpha_left = alpha_max_pml .* (1.0 .- normdist_left)
    K_left = 1.0 .+ (K_max_pml - 1.0) .* (normdist_left .^ npower)
    b_left = exp.(.-(d_left ./ K_left .+ alpha_left) .* dt)
    a_left = d_left .* (b_left .- 1.0) ./ (K_left .* (d_left .+ K_left .* alpha_left))
    b_K_left = b_left ./ K_left

    d_right = d0 .* (normdist_right .^ npower)
    alpha_right = alpha_max_pml .* (1.0 .- normdist_right)
    K_right = 1.0 .+ (K_max_pml - 1.0) .* (normdist_right .^ npower)
    b_right = exp.(.-(d_right ./ K_right .+ alpha_right) .* dt)
    a_right =
        d_right .* (b_right .- 1.0) ./
        (K_right .* (d_right .+ K_right .* alpha_right))
    b_K_right = b_right ./ K_right

    return a_left, a_right, b_K_left, b_K_right
end
