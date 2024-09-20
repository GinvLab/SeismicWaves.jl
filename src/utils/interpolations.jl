interpolate(m::Array{T, N}, interp_method) where {T, N} = collect(interp(interp_method, m, i) for i in 1:N)

interp(method, m::Array{<:Real, N}, dim::Int) where {N} = interp(method, m, [dim])
interp(method, m::Array{<:Real, N}, dims::Vector{Int}) where {N} = method(m, dims)

back_interp(method, m::Array{T, N}, ∂χ∂m_interp::Array{T, N}, dim::Int) where {T, N} = back_interp(method, m, ∂χ∂m_interp, [dim])
back_interp(method, m::Array{T, N}, ∂χ∂m_interp::Array{T, N}, dims::Vector{Int}) where {T, N} = begin
    res = zeros(T, size(m))
    for k in 1:method.width
        m_idxs = CartesianIndices(Tuple(
            i in dims ? (k:size(res, i)+k-method.width) : (1:size(res, i)) for i in 1:N
        ))
        res[m_idxs] .+= ∂χ∂m_interp .* ∂f∂m(method, m, k, dims)
    end
    return res
end

# Arithmetic average interpolation

struct ArithmeticAverageInterpolation <: InterpolationMethod
    width::Int
end

@views function (itp::ArithmeticAverageInterpolation)(m::Array{<:Real, N}, dims::Vector{Int}) where {N}
    return sum(
        m[CartesianIndices(Tuple(i in dims ? (j:size(m, i)+j-itp.width) : (1:size(m, i)) for i in 1:N))] for j in 1:itp.width
    ) ./ (itp.width^length(dims))
end

∂f∂m(itp::ArithmeticAverageInterpolation, m::Array{T, N}, _::Int, dims::Vector{Int}) where {T, N} = ones(T, size(m) .- Tuple(i in dims ? 1 : 0 for i in 1:N)) ./ (itp.width^length(dims))

# Armonic average interpolation

struct ArmonicAverageInterpolation <: InterpolationMethod
    width::Int
end

@views function (itp::ArmonicAverageInterpolation)(m::Array{<:Real, N}, dims::Vector{Int}) where {N}
    return (itp.width^length(dims)) ./ sum(
        1 ./ m[CartesianIndices(Tuple(i in dims ? (j:size(m, i)+j-itp.width) : (1:size(m, i)) for i in 1:N))] for j in 1:itp.width
    )
end

function ∂f∂m(itp::ArmonicAverageInterpolation, m::Array{T, N}, k::Int, dims::Vector{Int}) where {T, N}
    m_idxs = CartesianIndices(Tuple(
        i in dims ? (k:size(m, i)+k-itp.width) : (1:size(m, i)) for i in 1:N
    ))
    return (itp.width^length(dims)) ./
           sum(
        1 ./ m[CartesianIndices(Tuple(i in dims ? (j:size(m, i)+j-itp.width) : (1:size(m, i)) for i in 1:N))] for j in 1:itp.width
    ) .^ 2 .* m[m_idxs]
end