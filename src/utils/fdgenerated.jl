
"""
    fornberg(x::Vector{T}, m::Int) where {T}

Calculate the weights of a finite difference approximation of the `m`th derivative
with maximal order of accuracy at `0` using the nodes `x`, see
Fornberg (1998)
Calculation of Weights in Finite Difference Formulas
SIAM Rev. 40.3, pp. 685-691.
"""
function fornberg(x::SVector{O, T}, m::Int)::MVector{O, T} where {T, O}
    z = zero(T)
    n = length(x) - 1
    c = @MMatrix zeros(T, O, m+1)
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

function fdcoeffs_and_shifts(
    m::Int, vO::Val{O}, ::Val{T}
)::Tuple{NTuple{O, T}, NTuple{O, Int}} where {T, O}
    xs = SVector{O, T}(ntuple(i -> i - (O / 2 + 0.5), vO))
    coeffs = fornberg(xs, m)
    coeffs_tuple = ntuple(i -> coeffs[i], vO)
    shifts = ntuple(i -> iseven(O) ? i - (O ÷ 2) : i - (O ÷ 2) - 1, vO)
    return coeffs_tuple, shifts
end

@generated function generated_fdcoeffs_and_shifts(
    ::Val{P}, vO::Val{O}, vT::Val{T}
    )::Tuple{NTuple{O, T}, NTuple{O, Int}} where {P, T, O}
    coeffs, shifts = fdcoeffs_and_shifts(P, Val(O), Val(T))
    :($coeffs, $shifts)
end

# function get_coeffs_shifted_indices(
#     c::NTuple{O, T}, s::NTuple{O, Int},
#     i::Int, lb::Int, ub::Int, mlb::Bool, mub::Bool, half::Bool
#     )::Tuple{NTuple{O, T}, NTuple{O, Int}} where {T, O}
#     new_coeffs = ntuple(j -> lb <= i + s[j] <= ub ? c[j] : (i + s[j] < lb ? (!mlb ? zero(T) : (!half ? -c[j] : c[j]) ) : (!mub ? zero(T) : (!half ? -c[j] : c[j]))), Val(O))
#     shifted_indices = ntuple(j -> lb <= i + s[j] <= ub ? i + s[j] : (i + s[j] < lb ? (!mlb ? lb : (!half ? lb + (lb - (i + s[j])) : lb + (lb - (i + s[j])) - 1)) : (!mub ? ub : (!half ? ub - (i + s[j] - ub) : ub - (i + s[j] - ub) + 1))), Val(O))
#     return new_coeffs, shifted_indices
# end

Base.@propagate_inbounds function get_coeffs_shifted_indices(
    c::NTuple{O, T}, s::NTuple{O, Int},
    i::Int, lb::Int, ub::Int, mlb::Bool, mub::Bool, half::Bool
    )::Tuple{NTuple{O, T}, NTuple{O, Int}} where {T, O}

    new_coeffs = ntuple(j -> begin
                            if lb <= i + s[j] <= ub # inside bounds
                                c[j] # keep original coefficient
                            elseif (i + s[j] < lb && mlb) || (i + s[j] > ub && mub) # outside bounds, but mirror BCs
                                -c[j] # apply sign change for mirror BCs
                            else
                                zero(T) # outside bounds, no mirror BCs, set coefficient to zero
                            end
                        end, Val(O))

    shifted_indices = ntuple(j -> begin
                                if lb <= i + s[j] <= ub # inside bounds
                                    i + s[j] # keep original index
                                elseif i + s[j] < lb # below lower bound
                                    if mlb # mirror BCs at lower bound
                                        !half ? lb + (lb - (i + s[j])) : lb + (lb - (i + s[j])) - 1 # mirror index, if half grid, subtract 1 since we are mirroring around a grid point
                                    else # no mirror BCs at lower bound
                                        lb # clamp to lower bound, coefficient will be zero anyway
                                    end
                                elseif i + s[j] > ub # above upper bound
                                    if mub # mirror BCs at upper bound
                                        !half ? ub - ((i + s[j]) - ub) : ub - ((i + s[j]) - ub) + 1 # mirror index, if half grid, add 1 since we are mirroring around a grid point
                                    else # no mirror BCs at upper bound
                                        ub # clamp to upper bound, coefficient will be zero anyway
                                    end
                                end
                            end, Val(O))

    return new_coeffs, shifted_indices
end


"""
    ∂ᵐ(array, I, _Δ, vm, vp, vbdcheck; dir=1, lb=1, ub=size(array, dir)) where {T, N, A, m, p}

Compute the `m`th derivative of `array` in the direction `dir` at index `I` with inverse of grid spacing `_Δ` and order of accuracy `p`

# Arguments
- `array::A`: array of values to compute the derivative
- `I::NTuple{N, Int}`: index at which compute the derivative (central FD if `m` is even, right FD if `m` is odd)
- `_Δᵐ::T`: inverse of grid spacing to the m-th power
- `vm::Val{m}`: order of the derivative
- `p::Val{p}`: order of accuracy of the derivative
- `vbdcheck::Val{Bool}`: if `Val(true)` apply boundary checks, otherwise do not apply boundary checks

# Keyword Arguments
- `dir::Int=1`: direction in which compute the derivative
- `lb::Int=1`: lower bound index of the array in the direction of the derivative (in grid points)
- `ub::Int=size(array, dir)`: upper bound index of the array in the direction of the derivative (in grid points)
- `mlb::Bool=false`: if `true` apply mirror boundary conditions at the lower bound
- `mub::Bool=false`: if `true` apply mirror boundary conditions at the upper bound
- `half::Bool=false`: if `true` consider the array as points in the middle of the grid points
"""
Base.@propagate_inbounds function ∂ᵐ(
    array::A, I::NTuple{N, Int}, _Δᵐ::T, vm::Val{m}, ::Val{p}, ::Val{true};
    dir::Int=1, lb::Int=1, ub::Int=size(array, dir), mlb::Bool=false, mub::Bool=false, half::Bool=false
)::T where {T, N, A <: AbstractArray{T, N}, m, p}
    # Calculate number of grid points needed for stencil for `m`-th derivative of order `p`
    O = p + m - 1
    # Get the generated coefficients and shifts for the stencil
    coeffs, shifts = generated_fdcoeffs_and_shifts(vm, Val(O), Val(T))
    # Get the coefficients and shifted indices for the stencil with boundary checks
    coeffs_new, indices_new = get_coeffs_shifted_indices(coeffs, shifts, I[dir], lb, ub, mlb, mub, half)
    # Map the indices for the stencil in N-dimesions
    Is = ntuple(i -> ntuple(j -> j == dir ? indices_new[i] : I[j], Val(N)), Val(O))
    # Calculate the derivative using the stencil
    out = Ref(zero(eltype(array)))
    ntuple(Val(length(coeffs_new))) do i
        @inline 
        out[] += coeffs_new[i] * array[Is[i]...]
    end
    return  out[] * _Δᵐ
end

# No bounds checking
Base.@propagate_inbounds function ∂ᵐ(
    array::A, I::NTuple{N, Int}, _Δᵐ::T, vm::Val{m}, ::Val{p}, ::Val{false}; dir::Int=1
)::T where {T, N, A <: AbstractArray{T, N}, m, p}
    # Calculate number of grid points needed for stencil for `m`-th derivative of order `p`
    O = p + m - 1
    # Get the generated coefficients and shifts for the stencil
    coeffs, shifts = generated_fdcoeffs_and_shifts(vm, Val(O), Val(T))
    # Map the indices for the stencil in N-dimesions
    Is = ntuple(i -> ntuple(j -> j == dir ? I[dir] + shifts[i] : I[j], Val(N)), Val(O))
    # Calculate the derivative using the stencil
    out = Ref(zero(eltype(array)))
    ntuple(Val(length(coeffs))) do i
        @inline 
        out[] += coeffs[i] * array[Is[i]...]
    end
    return  out[] * _Δᵐ
end

# Scalar derivatives
Base.@propagate_inbounds ∂4th(x, I, _Δ, dir; kwargs...) = ∂ᵐ(x, I, _Δ, Val(1), Val(4), Val(true); dir=dir, kwargs...)
Base.@propagate_inbounds ∂x4th(x, I, _Δ; kwargs...) = ∂4th(x, I, _Δ, 1; kwargs...)
Base.@propagate_inbounds ∂y4th(x, I, _Δ; kwargs...) = ∂4th(x, I, _Δ, 2; kwargs...)
Base.@propagate_inbounds ∂z4th(x, I, _Δ; kwargs...) = ∂4th(x, I, _Δ, 3; kwargs...)
∂²4th(x, I, _Δ, dir; kwargs...) = ∂ᵐ(x, I, _Δ, Val(2), Val(4), Val(true); dir=dir, kwargs...)
∂²x4th(x, I, _Δ; kwargs...) = ∂²4th(x, I, _Δ, 1, kwargs...)
∂²y4th(x, I, _Δ; kwargs...) = ∂²4th(x, I, _Δ, 2, kwargs...)
∂²z4th(x, I, _Δ; kwargs...) = ∂²4th(x, I, _Δ, 3, kwargs...)

# Scalar derivatives with CPML damping
Base.@propagate_inbounds function ∂̃4th(x, ∂x, a, b, ψ, I, _Δ, dir, halo; half=false, kwargs...)
    ndim = size(x, dir)
    plusone = half ? 1 : 0
    idim = I[dir] + plusone
    iidim = I[dir] - (ndim - halo) + 1 + (halo + plusone)
    Iψ = ntuple(i -> i == dir ? idim : I[i], Val(length(I)))
    IIψ = ntuple(i -> i == dir ? iidim : I[i], Val(length(I)))
    # Apply CPML damping
    if idim <= (halo + plusone)
        ψ[Iψ...] = b[idim] * ψ[Iψ...] + a[idim] * ∂x
        ∂x + ψ[Iψ...]
    elseif idim >= ndim - halo
        ψ[IIψ...] = b[iidim] * ψ[IIψ...] + a[iidim] * ∂x
        ∂x + ψ[IIψ...]
    else
        ∂x
    end
end

Base.@propagate_inbounds ∂̃x4th(x, ∂x, a, b, ψ, I, _Δ, halo; half=false, kwargs...) = ∂̃4th(x, ∂x, a, b, ψ, I, _Δ, 1, halo; half=half, kwargs...)
Base.@propagate_inbounds ∂̃y4th(x, ∂x, a, b, ψ, I, _Δ, halo; half=false, kwargs...) = ∂̃4th(x, ∂x, a, b, ψ, I, _Δ, 2, halo; half=half, kwargs...)
Base.@propagate_inbounds ∂̃z4th(x, ∂x, a, b, ψ, I, _Δ, halo; half=false, kwargs...) = ∂̃4th(x, ∂x, a, b, ψ, I, _Δ, 3, halo; half=half, kwargs...)


function ∂̃²4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, dir, halo; half=false, kwargs...)
    ndim = size(x, dir)
    idim = I[dir]
    iidim = I[dir] - (ndim - halo) + 1 + halo
    Iψ = ntuple(i -> i == dir ? idim-1 : I[i], Val(length(I)))
    IIψ = ntuple(i -> i == dir ? iidim-1 : I[i], Val(length(I)))
    Iξ = ntuple(i -> i == dir ? idim : I[i], Val(length(I)))
    IIξ = ntuple(i -> i == dir ? iidim : I[i], Val(length(I)))
    # Apply CPML damping
    if idim <= halo
        ∂ψ = ∂4th(ψ, Iψ, _Δ, dir; half=half, kwargs...)
        ξ[Iξ...]  = b[idim] * ξ[Iξ...] + a[idim] * (∂²x + ∂ψ)
        ∂²x + ∂ψ + ξ[Iξ...]
    elseif idim >= ndim - halo + 1
        ∂ψ = ∂4th(ψ, IIψ, _Δ, dir; half=half, kwargs...)
        ξ[IIξ...] = b[iidim] * ξ[IIξ...] + a[iidim] * (∂²x + ∂ψ)
        ∂²x + ∂ψ + ξ[IIξ...]
    else
        ∂²x
    end
end

∂̃²x4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, halo; half=false, kwargs...) = ∂̃²4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, 1, halo; half=half, kwargs...)
∂̃²y4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, halo; half=false, kwargs...) = ∂̃²4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, 2, halo; half=half, kwargs...)
∂̃²z4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, halo; half=false, kwargs...) = ∂̃²4th(x, ∂²x, a, b, ψ, ξ, I, _Δ, 3, halo; half=half, kwargs...)
