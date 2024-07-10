
abstract type ElasticWaveSimulation{T, N} <: WaveSimulation{T, N} end

abstract type ElasticIsoWaveSimulation{T, N} <: ElasticWaveSimulation{T, N} end

abstract type ElasticMaterialProperties{T, N} <: MaterialProperties{T, N} end

abstract type AbstrElasticIsoMaterialProperties{T, N} <: ElasticMaterialProperties{T, N} end
