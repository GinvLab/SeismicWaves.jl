struct CPMLCoefficients{T<:Real}
    a_l::Vector{T}
    a_r::Vector{T}
    a_hl::Vector{T}
    a_hr::Vector{T}
    b_K_l::Vector{T}
    b_K_r::Vector{T}
    b_K_hl::Vector{T}
    b_K_hr::Vector{T}

    function CPMLCoefficients{T}(halo::Integer) where {T<:Real}
        new(
            zeros(T, halo),
            zeros(T, halo),
            zeros(T, halo+1),
            zeros(T, halo+1),
            zeros(T, halo),
            zeros(T, halo),
            zeros(T, halo+1),
            zeros(T, halo+1)
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
    alpha_max     = π*f0                            # CPML α multiplicative factor (half of dominating angular frequency)
    npower        = 2.0                             # CPML power coefficient
    K_max         = 1.0                             # CPML K coefficient value
    d0            = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness)     # damping profile
    a_l , a_r , b_K_l , b_K_r  = calc_Kab_CPML(halo,dt,npower,d0,alpha_max,K_max,"ongrd")
    a_hl, a_hr, b_K_hl, b_K_hr = calc_Kab_CPML(halo,dt,npower,d0,alpha_max,K_max,"halfgrd")

    cpmlcoeffs.a_l .= a_l
    cpmlcoeffs.a_r .= a_r
    cpmlcoeffs.a_hl .= a_hl
    cpmlcoeffs.a_hr .= a_hr
    cpmlcoeffs.b_K_l .= b_K_l
    cpmlcoeffs.b_K_r .= b_K_r
    cpmlcoeffs.b_K_hl .= b_K_hl
    cpmlcoeffs.b_K_hr .= b_K_hr

    println()
    @show halo
    @show size(a_l),size(a_r),size(a_hl),size(a_hr)
    @show size(b_K_l),size(b_K_r),size(b_K_hl),size(b_K_hr)
    
    return nothing
end


function calc_Kab_CPML(
    halo::Integer,
    dt::Float64,
    npower::Float64,
    d0::Float64,
    alpha_max_pml::Float64,
    K_max_pml::Float64,
    onwhere::String
    )
    
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
    dist = collect(LinRange(0-shift, Kab_size-shift-1, Kab_size))
    if onwhere == "halfgrd"
        dist[1] = 0
    end
    normdist_left = reverse(dist) ./ halo
    normdist_right = dist ./ halo

    d_left = d0 .* (normdist_left .^ npower)
    alpha_left = alpha_max_pml .* (1.0 .- normdist_left)
    K_left = 1.0 .+ (K_max_pml - 1.0) .* (normdist_left .^ npower)
    b_left = exp.( .-(d_left ./ K_left .+ alpha_left) .* dt )
    a_left = d_left .* (b_left .- 1.0) ./ (K_left .* (d_left .+ K_left .* alpha_left))
    b_K_left = b_left ./ K_left

    d_right = d0 .* (normdist_right .^ npower)
    alpha_right = alpha_max_pml .* (1.0 .- normdist_right)
    K_right = 1.0 .+ (K_max_pml - 1.0) .* (normdist_right .^ npower)
    b_right = exp.( .-(d_right ./ K_right .+ alpha_right) .* dt )
    a_right = d_right .* (b_right .- 1.0) ./ (K_right .* (d_right .+ K_right .* alpha_right))
    b_K_right = b_right ./ K_right

    return a_left, a_right, b_K_left, b_K_right
end
