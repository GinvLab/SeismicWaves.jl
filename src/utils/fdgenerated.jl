
"""
    fornberg(x::Vector{T}, m::Int) where {T}

Calculate the weights of a finite difference approximation of the `m`th derivative
with maximal order of accuracy at `0` using the nodes `x`, see
Fornberg (1998)
Calculation of Weights in Finite Difference Formulas
SIAM Rev. 40.3, pp. 685-691.
"""
function fornberg(xx::Vector{T}, m::Int) where {T}
    x = sort(xx)
    z = zero(T)
    n = length(x) - 1
    c = fill(zero(T), length(x), m+1)
    c1 = one(T)
    c4 = x[1] - z
    c[1,1] = one(T)
    for i in 1:n
        mn = min(i,m)
        c2 = one(T)
        c5 = c4
        c4 = x[i+1] - z
        for j in 0:i-1
            c3 = x[i+1] - x[j+1]
            c2 = c2 * c3
            if j == i-1
                for k in mn:-1:1
                    c[i+1,k+1] = c1 * (k*c[i,k]-c5*c[i,k+1]) / c2
                end
                c[i+1,1] = -c1*c5*c[i,1] / c2
            end
            for k in mn:-1:1
                c[j+1,k+1] = (c4*c[j+1,k+1]-k*c[j+1,k]) / c3
            end
            c[j+1,1] = c4*c[j+1,1] / c3
        end
        c1 = c2
    end

    c[:,end]
end

function fdcoeffs(deriv::Int, order::Int, ::Type{T})::Vector{T} where {T}
    nnn = order + deriv - 1
    xx = collect(T, (1:nnn) .- (nnn / 2 + 0.5))
    return fornberg(xx, deriv)
end

function fdshifts(deriv::Int, order::Int, dim::Int, N::Int)
    nnn = order + deriv - 1
    grid = collect(1:nnn) .- (nnn / 2 + 0.5)
    if (deriv+order) % 2 == 1
        grid .+= 0.5
    end
    grid = Int.(grid)
    shifts = Tuple( Tuple( i == dim && grid[j] != 0 ? grid[j] : 0 for i in 1:N) for j in 1:nnn )
    return shifts
end

@generated function fdcoeffs_gen(::Val{deriv}, ::Val{order}, ::Type{T}) where {deriv, order, T}
    Ncoeffs = order + deriv - 1
    coeffs = fdcoeffs(deriv, order, T)
    Core.println(coeffs)
    quote
        Base.@nexprs $Ncoeffs i -> a_i = $coeffs[i]
        Base.@ncall $Ncoeffs tuple a
    end
end

@generated function fdshifts_gen(::Val{deriv}, ::Val{order}, ::Val{dim}, ::Val{N}) where {deriv, order, dim, N}
    Ncoeffs = order + deriv - 1
    shifts = fdshifts(deriv, order, dim, N)
    Core.println(shifts)
    quote
        Base.@nexprs $Ncoeffs i -> a_i = $shifts[i]
        Base.@ncall $Ncoeffs tuple a
    end
end

@inline function ∂ⁿ_(
    array::A, I::NTuple{N, Int}, _Δ::T,
    vdim::Val{dim}, vderiv::Val{deriv}, vorder::Val{order}, vN::Val{N},
    ::Val{bdcheck}, ::Val{mirror_left}, ::Val{mirror_right}
)::T where {T, N, dim, deriv, order, A <: AbstractArray{T, N}, bdcheck, mirror_left, mirror_right}
    # Get generated coefficients and shifts
    coeffs = fdcoeffs_gen(vderiv, vorder, T)
    shifts = fdshifts_gen(vderiv, vorder, vdim, vN)

    if !bdcheck
        return sum(coeffs[i] * array[(I .+ shifts[i])...] for i in eachindex(coeffs)) * (_Δ ^ convert(T, deriv))
    else
        # Check if the stencil is within the bounds of the array
        Ileft = I[dim] + shifts[1][dim]
        Iright = I[dim] + shifts[end][dim]
        if 1 <= Ileft && Iright <= size(array, dim)
            return sum(coeffs[i] * array[(I .+ shifts[i])...] for i in eachindex(coeffs)) * (_Δ ^ convert(T, deriv))
        end
        # If not reduce the stencil to the valid points
        res = zero(T)
        for i in eachindex(coeffs)
            c = coeffs[i]
            s = shifts[i]
            # Check if current shift is within the bounds of the array
            if 1 <= I[dim] + s[dim] <= size(array, dim)
                res += c * array[(I .+ s)...]
            else
                # Check if the stencil is outside the array on the left
                if     I[dim] + s[dim] < 1
                    # If mirror_left is true, mirror the stencil
                    if mirror_left
                        Ishift_flipped = ntuple(i -> i == dim ? 1 - (I[dim] + s[dim]) : 0, N)
                        res += -c * array[(I .+ Ishift_flipped)...]

                    end
                # Check if the stencil is outside the array on the right
                elseif I[dim] + s[dim] > size(array, dim)
                    # If mirror_right is true, mirror the stencil
                    if mirror_right
                        Ishift_flipped = ntuple(i -> i == dim ? size(array, dim) - (I[dim] + s[dim]) : 0, N)
                        res += -c * array[(I .+ Ishift_flipped)...]
                    end
                end
            end
        end
        return res * (_Δ ^ convert(T, deriv))
    end
end

@inline function ∂ⁿ(
    array::A, I::NTuple{N, Int}, _Δ::T, dim::Int;
    deriv::Int=1, order::Int=2, bdcheck::Bool=false, mirror_left::Bool=false, mirror_right::Bool=false
)::T where {T, N, A <: AbstractArray{T, N}}
    return ∂ⁿ_(array, I, _Δ, Val(dim), Val(deriv), Val(order), Val(N), Val(bdcheck), Val(mirror_left), Val(mirror_right))
end
