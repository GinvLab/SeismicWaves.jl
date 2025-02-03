
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

function fdcoeffs(deriv::Int, order::Int)
    nnn = order + deriv - 1
    return fornberg(collect(1:(nnn)) .- (nnn / 2 + 0.5), deriv)
end

function fdidxs(deriv::Int, order::Int, dim::Int, I...)
    N = length(I)
    nnn = order + deriv - 1
    grid = collect(1:nnn) .- (nnn / 2 + 0.5)
    if (deriv+order) % 2 == 1
        grid .+= 0.5
    end
    grid = Int.(grid)
    Is = Tuple( Tuple(i == dim && grid[j] != 0 ? :($(I[i])+$(grid[j])) : I[i] for i in 1:N) for j in 1:nnn )
    return Is
end

function convert_float_literals(DT, ex)
    MacroTools.postwalk(x->typeof(x) <: AbstractFloat ? DT(x) : x, ex)
end

function ∂ⁿ_(A, dim::Int; I=(:i,), _Δ=1.0, deriv::Int=1, order::Int=2, bdcheck::Bool=true, mirror=(false, false), DataType=Float64)
    coeffs = fdcoeffs(deriv, order)
    Is = fdidxs(deriv, order, dim, I...)
    N = length(I)
    @assert N > 0 "Need at least one index!"
    mirror_left, mirror_right = mirror
    _Δ = deriv == 1 ? :( $_Δ ) : :( $_Δ ^ $deriv )
    # right boundary checks
    function right_checks_recursive(coeffs, Is, A, dim, _Δ, state)
        is = Tuple(Is[i][dim] for i in eachindex(Is))
        width = length(coeffs)
        # elseif in the checks
        if state > 1
            head = :elseif
        # zero if out of boundaries
        else
            return :(0.0)
        end
        condition = :(1 <= $(is[1]) <= $(is[state-1]) <= size($A, $dim) && $(is[state]) > size($A, $dim))
        if mirror_right
            # construct the mirrored stencil
            original_stencil = Expr(:call, :+, [:($(coeffs[i]) * $A[$(Is[i]...)]) for i in 1:state-1]...)                               # original stencil with indices up to boundary
            mirrored_Is = Tuple( Tuple(i == dim ? :($(Is[state-1][i])-$(j-state)) : Is[state-1][i] for i in 1:N) for j in state:width)  # indices mirrored around boundary
            mirrored_stencil = Expr(:call, :+, [:($(coeffs[i]) * -$A[$(mirrored_Is[i-state+1]...)]) for i in state:width]...)           # mirrored stencil with indices beyond boundary but mirrored
            stencil = Expr(:call, :*, Expr(:call, :+, original_stencil, mirrored_stencil), _Δ)                                          # sum the original and mirrored stencils     
        else
            stencil = Expr(:call, :*, Expr(:call, :+, [:($(coeffs[i]) * $A[$(Is[i]...)]) for i in 1:state-1]...), _Δ)
        end
        return Expr(head, condition, stencil, right_checks_recursive(coeffs, Is, A, dim, _Δ, state-1)) # recursively construct the if-elseif-else block
    end
    # (left) boundary checks
    function checks_recursive(coeffs, Is, A, dim, _Δ, state)
        is = Tuple(Is[i][dim] for i in eachindex(Is))
        width = length(coeffs)
        # first if in the checks
        if state == 1
            head = :if
        # elseif in the checks
        elseif state < width
            head = :elseif
        # time to check the right boundary
        else
            return right_checks_recursive(coeffs, Is, A, dim, _Δ, width)
        end
        condition = :($(is[state]) < 1 && 1 <= $(is[state+1]) <= $(is[end]) <= size($A, $dim))
        if mirror_left
            # construct the mirrored stencil
            original_stencil = Expr(:call, :+, [:($(coeffs[i]) * $A[$(Is[i]...)]) for i in state+1:width]...)                           # original stencil with indices beyond boundary
            mirrored_Is = Tuple( Tuple(i == dim ? :($(Is[state+1][i])+$(state-j)) : Is[state+1][i] for i in 1:N) for j in 1:state)          # indices mirrored around boundary
            mirrored_stencil = Expr(:call, :+, [:($(coeffs[i]) * -$A[$(mirrored_Is[i]...)]) for i in 1:state]...)                       # mirrored stencil with indices up to boundary but mirrored
            stencil = Expr(:call, :*, Expr(:call, :+, mirrored_stencil, original_stencil), _Δ)                                          # sum the original and mirrored stencils
        else
            stencil = Expr(:call, :*, Expr(:call, :+, [:($(coeffs[i]) * $A[$(Is[i]...)]) for i in state+1:width]...), _Δ)
        end
        return Expr(head, condition, stencil, checks_recursive(coeffs, Is, A, dim, _Δ, state+1))   # recursively construct the if-elseif-else block
    end
    # with boundary checks
    if bdcheck
        ileft = Is[1][dim]
        iright = Is[end][dim]
        expr = Expr(:if, :(1 <= $ileft && $iright <= size($A, $dim)),                                                           # if all indices inside range
                    Expr(:call, :*, Expr(:call, :+, [:($c * $A[$(Is[i]...)]) for (i, c) in enumerate(coeffs)]...), _Δ),         # inner case derivative
                    Expr(:block,                                                                                                # else
                        checks_recursive(coeffs, Is, A, dim, _Δ, 1)                                                             # boundary checks
                    )
                )
        return convert_float_literals(DataType, expr)
    end
    # without boundary checks
    return convert_float_literals(DataType, Expr(:call, :*, Expr(:call, :+, [:($c * $A[$(Is[i]...)]) for (i, c) in enumerate(coeffs)]...), _Δ))
end

function ∂̃_(A, a, b, ψ, dim::Int; I=(:i,), halo=:halo, halfgrid=true, DataType=Float64,  kwargs...)
    ∂Atemp = gensym(:∂A)
    iidim = gensym(:i)
    stencil = ∂ⁿ_(A, dim; I=I, deriv=1, kwargs...)
    plusone = halfgrid ? 0 : 1
    idim = :( $(I[dim]) + $plusone )
    ndim = :( size($A, $dim) + $plusone )
    Iψ  = [i == dim ?  idim : I[i] for i in eachindex(I)]
    IIψ = [i == dim ? iidim : I[i] for i in eachindex(I)]
    return convert_float_literals(DataType,
    quote
        $∂Atemp = $stencil
        if $idim <= ($halo + $plusone)
            $ψ[$(Iψ...)]  = $b[$idim ] * $ψ[$(Iψ...)] + $a[$idim ] * $∂Atemp
            $∂Atemp + $ψ[$(Iψ...)]
        elseif $idim >= $ndim - $halo
            $iidim = $idim - ($ndim - $halo) + 1 + ($halo + $plusone)
            $ψ[$(IIψ...)] = $b[$iidim] * $ψ[$(IIψ...)] + $a[$iidim] * $∂Atemp
            $∂Atemp + $ψ[$(IIψ...)]
        else
            $∂Atemp
        end
    end
    )
end

function ∂̃²_(A, a, b, ψ, ξ, dim::Int; I=(:i,), halo=:halo, DataType=Float64, kwargs...)
    ∂²Atemp = gensym(:∂²A)
    ∂ψtemp = gensym(:∂ψ)
    iidim = gensym(:i)
    Astencil       = ∂ⁿ_(A, dim; I=I, deriv=2, kwargs...)
    idim = :( $(I[dim]) )
    ndim = :( size($A, $dim) )
    Iψ  = [i == dim ?  :($idim - 1) : I[i] for i in eachindex(I)]
    IIψ = [i == dim ?  :($iidim - 1) : I[i] for i in eachindex(I)]
    Iξ  = [i == dim ?  idim : I[i] for i in eachindex(I)]
    IIξ = [i == dim ? iidim : I[i] for i in eachindex(I)]
    ψstencil_left  = ∂ⁿ_(ψ, dim; I=Iψ, deriv=1, kwargs..., bdcheck=false)
    ψstencil_right = ∂ⁿ_(ψ, dim; I=IIψ, deriv=1, kwargs..., bdcheck=false)
    return convert_float_literals(DataType,
    quote
        $∂²Atemp = $Astencil
        if $idim <= $halo
            $∂ψtemp = $ψstencil_left
            $ξ[$(Iξ...)]  = $b[$idim ] * $ξ[$(Iξ...)] + $a[$idim ] * ($∂²Atemp + $∂ψtemp)
            $∂²Atemp + $∂ψtemp + $ξ[$(Iξ...)]
        elseif $idim >= $ndim - $halo + 1
            $iidim = $idim - ($ndim - $halo) + 1 + $halo
            $∂ψtemp = $ψstencil_right
            $ξ[$(IIξ...)] = $b[$iidim] * $ξ[$(IIξ...)] + $a[$iidim] * ($∂²Atemp + $∂ψtemp)
            $∂²Atemp + $∂ψtemp + $ξ[$(IIξ...)]
        else
            $∂²Atemp
        end
    end
    )
end

function ∇ⁿ_(args...; I=(), _Δ=(), kwargs...)
    N = length(I)
    @assert N == length(_Δ)
    return Expr(:tuple, (∂ⁿ_(length(args) == 1 ? args[1] : args[i], i; _Δ=_Δ[i], I=I, kwargs...) for i in eachindex(_Δ))...)
end

function ∇̃_(args...; I=(), _Δ=(), kwargs...)
    N = length(I)
    @assert N == length(_Δ)
    A = args[1]
    cpml_args = collect(args[1+(1+3*(i-1)):1+3*(i-1)+3] for i in 1:N)
    return Expr(:tuple, (∂̃_(A, cpml_args[i]..., i; _Δ=_Δ[i], I=I, kwargs..., deriv=1) for i in eachindex(_Δ))...)
end

function ∇̃²_(args...; I=(), _Δ=(), kwargs...)
    N = length(I)
    @assert N == length(_Δ)
    A = args[1]
    cpml_args = collect(args[1+(1+4*(i-1)):1+4*(i-1)+4] for i in 1:N)
    return Expr(:tuple, (∂̃²_(A, cpml_args[i]..., i; _Δ=_Δ[i], I=I, kwargs...) for i in eachindex(_Δ))...)
end

function extract_kwargs(args...)
    kwargs = Dict{Symbol, Any}()
    positional = []
    for arg in args
        if isa(arg, Expr) && arg.head == :(=) && length(arg.args) == 2 && isa(arg.args[1], Symbol)
            rightarghead = isa(arg.args[2], Expr) ? arg.args[2].head : nothing
            if rightarghead == :tuple
                kwargs[arg.args[1]] = Tuple(arg.args[2].args)
            elseif arg.args[1] == :DataType
                if arg.args[2] == :Float32
                    kwargs[arg.args[1]] = Float32
                elseif arg.args[2] == :Float64
                    kwargs[arg.args[1]] = Float64
                else
                    error("Only Float32 and Float64 are supported")
                end
            else
                kwargs[arg.args[1]] = arg.args[2]
            end
        else
            push!(positional, arg)
        end
    end
    return positional, kwargs
end

# SCALAR DERIVATIVES
macro ∂(  args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs...;    kwargs..., deriv=1)           ) end
macro ∂²( args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs...;    kwargs..., deriv=2)           ) end
macro ∂ⁿ( args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs...;    kwargs...)                    ) end
macro ∂x( args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 1; kwargs..., deriv=1)           ) end
macro ∂y( args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 2; kwargs..., deriv=1)           ) end
macro ∂z( args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 3; kwargs..., deriv=1)           ) end
macro ∂²x(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 1; kwargs..., deriv=2)           ) end
macro ∂²y(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 2; kwargs..., deriv=2)           ) end
macro ∂²z(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 3; kwargs..., deriv=2)           ) end
macro ∂ⁿx(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 1; kwargs...)                    ) end
macro ∂ⁿy(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 2; kwargs...)                    ) end
macro ∂ⁿz(args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∂ⁿ_(posargs..., 3; kwargs...)                    ) end

# VECTOR DERIVATIVES (GRADIENT, DIVERGENCE, LAPLACIAN)
macro ∇(  args...)  posargs, kwargs = extract_kwargs(args...); esc(                  ∇ⁿ_(posargs...;    kwargs...)                    ) end
macro div(args...)  posargs, kwargs = extract_kwargs(args...); esc( Expr(:call, :+,  ∇ⁿ_(posargs...;    kwargs..., deriv=1).args... ) ) end
macro ∇²( args...)  posargs, kwargs = extract_kwargs(args...); esc( Expr(:call, :+,  ∇ⁿ_(posargs...;    kwargs..., deriv=2).args... ) ) end

# SCALAR DERIVATIVES WITH CPML DAMPING
macro ∂̃( args...)  posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃_(posargs...;     kwargs...)                    ) end
macro ∂̃x( args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃_(posargs..., 1;  kwargs...)                    ) end
macro ∂̃y( args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃_(posargs..., 2;  kwargs...)                    ) end
macro ∂̃z( args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃_(posargs..., 3;  kwargs...)                    ) end
macro ∂̃²( args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃²_(posargs...;    kwargs...)                    ) end
macro ∂̃²x(args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃²_(posargs..., 1; kwargs...)                    ) end
macro ∂̃²y(args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃²_(posargs..., 2; kwargs...)                    ) end
macro ∂̃²z(args...) posargs, kwargs = extract_kwargs(args...); esc(                   ∂̃²_(posargs..., 3; kwargs...)                    ) end

# VECTOR DERIVATIVES (GRADIENT, DIVERGENCE, LAPLACIAN) WITH CPML DAMPING
macro ∇̃( args...)  posargs, kwargs = extract_kwargs(args...); esc(                   ∇̃_(posargs...;    kwargs...)                     ) end
macro diṽ(args...) posargs, kwargs = extract_kwargs(args...); esc( Expr(:call, :+,   ∇̃_(posargs...;    kwargs...).args... )           ) end
macro ∇̃²( args...) posargs, kwargs = extract_kwargs(args...); esc( Expr(:call, :+,   ∇̃²_(posargs...;   kwargs...).args... )           ) end