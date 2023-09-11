@doc raw"""
    rickersource1D(t::Real, t0::Real, f0::Real)    

Ricker source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
rickersource1D(t::Real, t0::Real, f0::Real)::Real = (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-(pi * f0 * (t - t0))^2)

@doc raw"""
    gaussource1D(t::Real, t0::Real, f0::Real)

Gaussian source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
gaussource1D(t::Real, t0::Real, f0::Real)::Real = -(pi * f0 * (t - t0))^2 * exp(-(pi * f0 * (t - t0))^2)

@doc raw"""
    interp_avg(a::Array{<:Real, N}, dim)

Interpolate dimension `dim` of array `a` using arithmetic average.
"""
@views function interp_avg(a::Array{<:Real, N}, dim) where {N}
    left = A[CartesianIndices(Tuple( i == dim ? (1:size(a,i)-1) : (1:size(a,i)) for i in 1:N ))]
    right = A[CartesianIndices(Tuple( i == dim ? (2:size(a,i)) : (1:size(a,i)) for i in 1:N ))]
    return (left .+ right) .* 0.5
end