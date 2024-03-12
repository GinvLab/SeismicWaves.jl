
abstract type ElasticWaveSimul{N} <: WaveSimul{N} end

abstract type ElasticIsoWaveSimul{N} <: ElasticWaveSimul{N} end

abstract type ElasticMaterialProperties{N} <: MaterialProperties{N} end

abstract type AbstrElasticIsoMaterialProperties{N} <: ElasticMaterialProperties{N} end
