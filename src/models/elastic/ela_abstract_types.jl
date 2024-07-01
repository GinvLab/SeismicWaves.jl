
abstract type ElasticWaveSimul{N} <: WaveSimul{N} end

abstract type ElasticIsoWaveSimul{N} <: ElasticWaveSimul{N} end

abstract type ElasticMaterialProperties{T, N} <: MaterialProperties{T, N} end

abstract type AbstrElasticIsoMaterialProperties{T, N} <: ElasticMaterialProperties{T, N} end
