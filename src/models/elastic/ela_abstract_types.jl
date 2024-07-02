
abstract type ElasticWaveSimul{T,N} <: WaveSimul{T,N} end

abstract type ElasticIsoWaveSimul{T,N} <: ElasticWaveSimul{T,N} end

abstract type ElasticMaterialProperties{T, N} <: MaterialProperties{T, N} end

abstract type AbstrElasticIsoMaterialProperties{T, N} <: ElasticMaterialProperties{T, N} end
