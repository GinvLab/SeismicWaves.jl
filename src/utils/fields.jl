mutable struct ScalarConstantField{T} <: AbstractField{T}
    value::T
end

mutable struct ScalarVariableField{T, N, A <: AbstractArray{T, N}} <: AbstractField{T}
    value::A
end

mutable struct MultiConstantField{T, N} <: AbstractField{T}
    value::Vector{T}
    MultiConstantField(value::NTuple{N, T}) where {T, N} = new{T, N}(collect(value))
end

mutable struct MultiVariableField{T, N, A <: AbstractArray{T, N}} <: AbstractField{T}
    value::Vector{A}
end

Base.copyto!(dest::ScalarConstantField, src::ScalarConstantField) = (dest.value = src.value)
Base.copyto!(dest::ScalarVariableField, src::ScalarConstantField) = (dest.value .= src.value)
Base.copyto!(dest::ScalarVariableField, src::ScalarVariableField) = copyto!(dest.value, src.value)
Base.copyto!(dest::MultiConstantField, src::Union{ScalarConstantField, MultiConstantField}) = (dest.value .= src.value)
Base.copyto!(dest::MultiVariableField, src::ScalarConstantField) = begin
    for vdest in dest.value
        vdest .= src.value
    end
end
Base.copyto!(dest::MultiVariableField, src::MultiConstantField) = begin
    for (vdest, vsrc) in zip(dest.value, src.value)
        vdest .= vsrc
    end
end
Base.copyto!(dest::MultiVariableField, src::MultiVariableField) = begin
    for (vdest, vsrc) in zip(dest.value, src.value)
        copyto!(vdest, vsrc)
    end
end


setzero!(field::ScalarConstantField{T}) where {T} = (field.value = zero(T); return field)
setzero!(field::MultiConstantField{T}) where {T} = (field.value .= zero(T); return field)
setzero!(field::ScalarVariableField{T}) where {T} = (field.value .= zero(T); return field)
setzero!(field::MultiVariableField{T}) where {T} = begin
    for vfield in field.value
        vfield .= zero(T)
    end
    return field
end

Base.zero(::Type{ScalarConstantField{T}}) where {T} = ScalarConstantField(zero(T))
Base.zero(::Type{MultiConstantField{T, N}}) where {T, N} = MultiConstantField(ntuple(_ -> zero(T), N))
Base.zero(::ScalarConstantField{T}) where {T} = zero(ScalarConstantField{T})
Base.zero(::MultiConstantField{T, N}) where {T, N} = zero(MultiConstantField{T, N})
Base.zero(field::ScalarVariableField{T, N, A}) where {T, N, A} = ScalarVariableField(zero(field.value))
Base.zero(field::MultiVariableField{T, N, A}) where {T, N, A} = MultiVariableField([zero(f) for f in field.value])