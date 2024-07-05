module SeismicWaves_CUDABackendExt

using SeismicWaves, CUDA
using SeismicWaves: CPMLBoundaryCondition, LocalGrid, AcousticCDCPMLWaveSimulation, AcousticVDStaggeredCPMLWaveSimulation

# Include CUDA backends
include("../src/models/acoustic/backends/Acoustic1D_CD_CPML_CUDA.jl")
include("../src/models/acoustic/backends/Acoustic2D_CD_CPML_CUDA.jl")
include("../src/models/acoustic/backends/Acoustic3D_CD_CPML_CUDA.jl")
include("../src/models/acoustic/backends/Acoustic1D_VD_CPML_CUDA.jl")
include("../src/models/acoustic/backends/Acoustic2D_VD_CPML_CUDA.jl")
include("../src/models/elastic/backends/Elastic2D_Iso_CPML_CUDA.jl")

# Overload backend selection
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:CUDA}}) = Acoustic1D_CD_CPML_CUDA
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:CUDA}}) = Acoustic2D_CD_CPML_CUDA
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 3}}, ::Type{Val{:CUDA}}) = Acoustic3D_CD_CPML_CUDA
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:CUDA}}) = Acoustic1D_VD_CPML_CUDA
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:CUDA}}) = Acoustic2D_VD_CPML_CUDA
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:CUDA}}) = Elastic2D_Iso_CPML_CUDA
end