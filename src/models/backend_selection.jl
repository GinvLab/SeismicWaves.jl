
select_backend(wavesim_type::Type{<:WaveSimulation}, parall::Symbol) =
    select_backend(BoundaryConditionTrait(wavesim_type), GridTrait(wavesim_type), wavesim_type, Val{parall})

function select_backend(
    ::BoundaryConditionTrait,
    ::GridTrait,
    wavesim_type::Type{<:WaveSimulation},
    ::Type{Val{parall}}
) where {parall}
    parasym = [:serial, :threads, :GPU, :threadpersrc]
    error(
        "No backend found for model of type $(wavesim_type) and `parall =` $(parall). Argument `parall` must be one of the following symbols: $parasym."
    )
end

## Remark:
## CD = constant density
## VD = variable density

# Backend selections for AcousticCDCPMLWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 1}}, ::Type{Val{:serial}}) = Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 2}}, ::Type{Val{:serial}}) = Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 3}}, ::Type{Val{:serial}}) = Acoustic3D_CD_CPML_Serial

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 1}}, ::Type{Val{:threads}}) = Acoustic1D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 2}}, ::Type{Val{:threads}}) = Acoustic2D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 3}}, ::Type{Val{:threads}}) = Acoustic3D_CD_CPML_Threads

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 1}}, ::Type{Val{:threadpersrc}}) = Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 2}}, ::Type{Val{:threadpersrc}}) = Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{Float64, 3}}, ::Type{Val{:threadpersrc}}) = Acoustic3D_CD_CPML_Serial

# Backend selections for AcousticVDStaggeredCPMLWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{Float64, 1}}, ::Type{Val{:threads}}) = Acoustic1D_VD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticVDStaggeredCPMLWaveSimul{Float64, 2}}, ::Type{Val{:threads}}) = Acoustic2D_VD_CPML_Threads

# Backend selections for ElasticIsoWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:ElasticIsoWaveSimul{Float64, 2}}, ::Type{Val{:serial}}) = Elastic2D_Iso_CPML_Serial
