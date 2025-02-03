module SeismicWaves_AMDGPUBackendExt

using SeismicWaves, AMDGPU
using SeismicWaves: CPMLBoundaryCondition, LocalGrid, AcousticCDCPMLWaveSimulation, AcousticVDStaggeredCPMLWaveSimulation, ElasticIsoCPMLWaveSimulation

# Include AMDGPU backends
include("../src/models/acoustic/backends/Acoustic1D_CD_CPML_AMDGPU.jl")
include("../src/models/acoustic/backends/Acoustic2D_CD_CPML_AMDGPU.jl")
include("../src/models/acoustic/backends/Acoustic3D_CD_CPML_AMDGPU.jl")
include("../src/models/acoustic/backends/Acoustic1D_VD_CPML_AMDGPU.jl")
include("../src/models/acoustic/backends/Acoustic2D_VD_CPML_AMDGPU.jl")
include("../src/models/elastic/backends/Elastic2D_Iso_CPML_AMDGPU.jl")

# Overload backend selection
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:AMDGPU}}) =
    Acoustic1D_CD_CPML_AMDGPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:AMDGPU}}) =
    Acoustic2D_CD_CPML_AMDGPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 3}}, ::Type{Val{:AMDGPU}}) =
    Acoustic3D_CD_CPML_AMDGPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 1}}, ::Type{Val{:AMDGPU}}) =
    Acoustic1D_VD_CPML_AMDGPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:AMDGPU}}) =
    Acoustic2D_VD_CPML_AMDGPU
SeismicWaves.select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoCPMLWaveSimulation{<:Union{Float16, Float32, Float64}, 2}}, ::Type{Val{:AMDGPU}}) =
    Elastic2D_Iso_CPML_AMDGPU

end