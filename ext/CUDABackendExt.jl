module CUDABackendExt

using SeismicWaves, CUDA
using SeismicWaves: CPMLBoundaryCondition, LocalGrid, AcousticCDCPMLWaveSimulation, AcousticVDStaggeredCPMLWaveSimulation

# Include CUDA backends
include("../src/models/acoustic/backends/Acoustic1D_CD_CPML_GPU.jl")
include("../src/models/acoustic/backends/Acoustic2D_CD_CPML_GPU.jl")
include("../src/models/acoustic/backends/Acoustic3D_CD_CPML_GPU.jl")
include("../src/models/acoustic/backends/Acoustic1D_VD_CPML_GPU.jl")
include("../src/models/acoustic/backends/Acoustic2D_VD_CPML_GPU.jl")

# Overload backend selection
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:GPU}}) = Acoustic1D_CD_CPML_GPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:GPU}}) = Acoustic2D_CD_CPML_GPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{Union{Float16, Float32, Float64}, 3}}, ::Type{Val{:GPU}}) = Acoustic3D_CD_CPML_GPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:GPU}}) = Acoustic1D_VD_CPML_GPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:GPU}}) = Acoustic2D_VD_CPML_GPU

end