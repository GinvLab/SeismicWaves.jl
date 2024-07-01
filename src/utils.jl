@doc """
$(TYPEDSIGNATURES)

Ricker source (second derivative of gaussian) source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
rickerstf(t::Real, t0::Real, f0::Real)::Real = (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-(pi * f0 * (t - t0))^2)

@doc """
$(TYPEDSIGNATURES) 

First derivative of gaussian source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
gaussderivstf(t::Real, t0::Real, f0::Real)::Real = (t - t0) * exp(-(pi * f0 * (t - t0))^2)

@doc """
$(TYPEDSIGNATURES) 

Gaussian source time function for current time `t`, activation time `t0` and dominating frequency `f0`.
"""
gaussstf(t::Real, t0::Real, f0::Real)::Real = -exp(-(pi * f0 * (t - t0))^2) / (2 * (pi * f0)^2)

struct ArithmeticAverageInterpolation <: InterpolationMethod
    width::Int
end

interpolate(a::Array{T, N}, interp_method) where {T, N} = collect(interp(interp_method, a, i) for i in 1:N)

@views function interp(method::ArithmeticAverageInterpolation, a::Array{<:Real, N}, dim) where {N}
    return sum(
        a[CartesianIndices(Tuple(i == dim ? (j:size(a, i)+j-method.width) : (1:size(a, i)) for i in 1:N))] for j in 1:method.width
    ) ./ method.width
end

"""
$(TYPEDSIGNATURES) 

Compute an optimal distribution of tasks (nsrc) for a given number of workers (threads).
Returns a vector of UnitRange object.
"""
function distribsrcs(nsrc::Integer, nw::Integer)
    ## calculate how to subdivide the srcs among the workers
    if nsrc >= nw
        dis = div(nsrc, nw)
        grpsizes = dis * ones(Int64, nw)
        resto = mod(nsrc, nw)
        if resto > 0
            ## add the reminder
            grpsizes[1:resto] .+= 1
        end
    else
        ## if more workers than sources use only necessary workers
        grpsizes = ones(Int64, nsrc)
    end
    ## now set the indices for groups of srcs
    grpsrc = UnitRange.(cumsum(grpsizes) .- grpsizes .+ 1, cumsum(grpsizes))
    return grpsrc
end
