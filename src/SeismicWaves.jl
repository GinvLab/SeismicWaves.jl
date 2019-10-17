

module SeismicWaves


export gaussource1D,rickersource1D

include("AcousticWav/AcousticWaves.jl")
using .AcousticWaves
#export f1


include("ElasticWav/ElasticWaves.jl")
using .ElasticWaves
#export f2

#
include("utils.jl")

end # module
