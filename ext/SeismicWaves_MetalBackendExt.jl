module SeismicWaves_MetalBackendExt

using SeismicWaves, Metal
using SeismicWaves: CPMLBoundaryCondition, LocalGrid, AcousticCDCPMLWaveSimulation, AcousticVDStaggeredCPMLWaveSimulation, ElasticIsoCPMLWaveSimulation

# Include Metal backends
include("../src/models/acoustic/backends/Acoustic1D_CD_CPML_Metal.jl")
include("../src/models/acoustic/backends/Acoustic2D_CD_CPML_Metal.jl")
include("../src/models/acoustic/backends/Acoustic3D_CD_CPML_Metal.jl")
include("../src/models/acoustic/backends/Acoustic1D_VD_CPML_Metal.jl")
include("../src/models/acoustic/backends/Acoustic2D_VD_CPML_Metal.jl")
include("../src/models/elastic/backends/Elastic2D_Iso_CPML_Metal.jl")

# Overload backend selection
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Float32, 1}}, ::Type{Val{:Metal}}) =
    Acoustic1D_CD_CPML_Metal
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Float32, 2}}, ::Type{Val{:Metal}}) =
    Acoustic2D_CD_CPML_Metal
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Float32, 3}}, ::Type{Val{:Metal}}) =
    Acoustic3D_CD_CPML_Metal
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{Float32, 1}}, ::Type{Val{:Metal}}) =
    Acoustic1D_VD_CPML_Metal
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{Float32, 2}}, ::Type{Val{:Metal}}) =
    Acoustic2D_VD_CPML_Metal
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoCPMLWaveSimulation{Float32, 2}}, ::Type{Val{:Metal}}) =
    Elastic2D_Iso_CPML_Metal
end