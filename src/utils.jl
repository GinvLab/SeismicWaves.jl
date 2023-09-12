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

struct ArithmeticAverageInterpolation <: InterpolationMethod
    width::Int
end

@views function interp(method::ArithmeticAverageInterpolation, a::Array{<:Real, N}, dim) where {N}
    return sum(
        a[CartesianIndices(Tuple(i == dim ? (j:size(a, i)+j-method.width) : (1:size(a, i)) for i in 1:N))] for j in 1:method.width
    ) ./ method.width
end

@views function jacobian(method::ArithmeticAverageInterpolation, a::Array{<:Real, N}, dim) where {N}
    return fill(1, method.width) ./ method.width
end