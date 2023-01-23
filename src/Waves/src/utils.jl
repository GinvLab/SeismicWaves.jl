@doc raw"""
    rickersource1D(t::Real, t0::Real, f0::Real)    

Ricker source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
function rickersource1D(t::Real, t0::Real, f0::Real)
    return (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-((pi * f0 * (t - t0))^2))
end