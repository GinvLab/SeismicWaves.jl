struct ScalarConstantField{N, T} <: AbstractField{N, T}
    value::T
end

struct ScalarVariableField{N, T, A <: AbstractArray{T}} <: AbstractField{N, T}
    value::A
end

struct MultiConstantField{N, T, V <: AbstractArray{T}} <: AbstractField{N, T}
    value::V
end

struct MultiVariableField{N, T, A <: AbstractArray{T}, V <: AbstractArray{A}} <: AbstractField{N, T}
    value::V
end

function SeismicWaves.copyto!(dest::ScalarVariableField{N, T, A}, src::ScalarConstantField{N, T}) where {N, T, A <: AbstractArray{T}}
    dest.value .= src.value
end
SeismicWaves.copyto!(dest::ScalarVariableField{N, T, A}, src::ScalarVariableField{N, T, A}) where {N, T, A <: AbstractArray{T}} = copyto!(dest.value, src.value)
function SeismicWaves.copyto!(dest::MultiVariableField{N,T,A,V}, src::ScalarConstantField{N,T}) where {N, T, A <: AbstractArray{T}, V <: AbstractArray{A}}
    for vdest in dest.value
        vdest .= src.value
    end
end
function SeismicWaves.copyto!(dest::MultiVariableField{N,T,A,V1}, src::MultiConstantField{N,T,V2}) where {N, T, A <: AbstractArray{T}, V1 <: AbstractArray{A}, V2 <: AbstractArray{T}}
    for (vdest, vsrc) in zip(dest.value, src.value)
        vdest .= vsrc
    end
end
function SeismicWaves.copyto!(dest::MultiVariableField{N,T,A,V}, src::MultiVariableField{N,T,A,V}) where {N, T, A <: AbstractArray{T}, V <: AbstractArray{A}}
    for (vdest, vsrc) in zip(dest.value, src.value)
        copyto!(vdest, vsrc)
    end
end