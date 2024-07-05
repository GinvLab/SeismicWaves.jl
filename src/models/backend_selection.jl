
select_backend(wavesim_type::Type{<:WaveSimulation}, parall::Symbol) =
    select_backend(BoundaryConditionTrait(wavesim_type), GridTrait(wavesim_type), wavesim_type, Val{parall})

function select_backend(
    ::BoundaryConditionTrait,
    ::GridTrait,
    wavesim_type::Type{<:WaveSimulation},
    ::Type{Val{parall}}
) where {parall}
    parasym = [:serial, :threads, :CUDA, :AMDGPU, :threadpersrc]
    error(
        "No backend found for model of type $(wavesim_type) and `parall =` $(parall). Argument `parall` must be one of the following symbols: $parasym."
    )
end

## Remark:
## CD = constant density
## VD = variable density

# Backend selections for AcousticCDCPMLWaveSimulation
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 1}}, ::Type{Val{:serial}}) = Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:serial}}) = Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 3}}, ::Type{Val{:serial}}) = Acoustic3D_CD_CPML_Serial

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 1}}, ::Type{Val{:threads}}) = Acoustic1D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:threads}}) = Acoustic2D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 3}}, ::Type{Val{:threads}}) = Acoustic3D_CD_CPML_Threads

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 1}}, ::Type{Val{:threadpersrc}}) = Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:threadpersrc}}) = Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimulation{<:AbstractFloat, 3}}, ::Type{Val{:threadpersrc}}) = Acoustic3D_CD_CPML_Serial

# Backend selections for AcousticVDStaggeredCPMLWaveSimulation
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:AbstractFloat, 1}}, ::Type{Val{:threads}}) = Acoustic1D_VD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:threads}}) = Acoustic2D_VD_CPML_Threads

# Backend selections for ElasticIsoWaveSimulation
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:serial}}) = Elastic2D_Iso_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoWaveSimulation{<:AbstractFloat, 2}}, ::Type{Val{:threads}}) = Elastic2D_Iso_CPML_Threads
