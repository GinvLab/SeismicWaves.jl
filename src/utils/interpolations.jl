interpolate(m::Array{T, N}, interp_method::AbstractInterpolationMethod) where {T, N} = collect(interp(interp_method, m, i) for i in 1:N)

interp(method::AbstractInterpolationMethod, m::Array{<:Real, N}, dim::Int) where {N} = interp(method, m, [dim])
interp(method::AbstractInterpolationMethod, m::Array{<:Real, N}, dims::Vector{Int}) where {N} = method(m, dims)

function binary_permutations(N::Int)
    permutations = []
    for i in 0:(2^N-1)
        binary_str = bitstring(i)[end-N+1:end]
        push!(permutations, [parse(Int, c) for c in binary_str])
    end
    return permutations
end

back_interp(method, m::Array{T, N}, ∂χ∂m_interp::Array{T, N}, dim::Int) where {T, N} = back_interp(method, m, ∂χ∂m_interp, [dim])
back_interp(method, m::Array{T, N}, ∂χ∂m_interp::Array{T, N}, dims::Vector{Int}) where {T, N} = begin
    res = zeros(T, size(m))
    perms = binary_permutations(length(dims))
    for p in perms
        pp = zeros(Int, N)
        pp[dims] .= p
        idxs = CartesianIndices(Tuple(
            i in dims ? (1:size(m, i)-1) .+ pp[i] : (1:size(m, i)) for i in 1:N
        ))
        res[idxs] .+= ∂χ∂m_interp .* ∂f∂m(method, m, idxs, dims)
    end
    return res
end

# Arithmetic average interpolation
struct ArithmeticAverageInterpolation <: AbstractInterpolationMethod end
# Harmonic average interpolation
struct HarmonicAverageInterpolation <: AbstractInterpolationMethod end

function (itp::ArithmeticAverageInterpolation)(m::Array{<:Real, N}, dims::Vector{Int}) where {N}
    itp_obj = Interpolations.interpolate(m, BSpline(Linear()))
    return itp_obj([i in dims ? (1.5:size(m, i)-0.5) : (1:size(m, i)) for i in 1:N]...)
end

function (itp::HarmonicAverageInterpolation)(m::Array{<:Real, N}, dims::Vector{Int}) where {N}
    itp_obj = Interpolations.interpolate(1 ./ m, BSpline(Linear()))
    return 1 ./ itp_obj([i in dims ? (1.5:size(m, i)-0.5) : (1:size(m, i)) for i in 1:N]...)
end

∂f∂m(_::ArithmeticAverageInterpolation, m::Array{T, N}, idxs, dims::Vector{Int}) where {T, N} = ones(T, size(m) .- Tuple(i in dims ? 1 : 0 for i in 1:N)) ./ (2^length(dims))

∂f∂m(itp::HarmonicAverageInterpolation, m::Array{T, N}, idxs, dims::Vector{Int}) where {T, N} = (itp(m, dims) .^ 2) ./ (m[idxs] .^ 2) ./ (2^length(dims))