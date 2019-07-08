

module SeismicWaves


include("AcousticWav/AcousticWaves.jl")
using .AcousticWaves
export f1


include("ElasticWav/ElasticWaves.jl")
using .ElasticWaves
export f2



end # module
