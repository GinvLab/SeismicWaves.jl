
###########################################################

struct AcousticCDCPMLWaveSimul{N} <: AcousticCDWaveSimul{N}
    # Physics
    ls::NTuple{N, <:Integer}
    # Numerics
    ns::NTuple{N, <:Integer}
    gridspacing::NTuple{N, <:Real}
    nt::Integer
    dt::Real
    # BDC and CPML parameters
    halo::Integer
    rcoef::Real
    freetop::Bool
    # Snapshots paramters
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{<:Array{<:Real}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Arrays
    vel::Array{<:Real, N}
    fact::Array{<:Real, N}
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Backend
    backend::Module

    function AcousticCDCPMLWaveSimul{N}(
        ns::NTuple{N, <:Integer},
        gridspacing::NTuple{N, <:Real},
        nt::Integer,
        dt::Real,
        halo::Integer,
        rcoef::Real,
        parall::Symbol;
        freetop::Bool=true,
        snapevery::Union{<:Integer, Nothing}=nothing,
        infoevery::Union{<:Integer, Nothing}=nothing
    ) where {N}
        # Check numerics
        @assert all(ns .> 0) "All numbers of grid points must be positive!"
        @assert all(gridspacing .> 0) "All cell sizes must be positive!"
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"

        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        ns_cpml = freetop ? ns[1:(end-1)] : ns
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Compute model sizes
        ls = gridspacing .* (ns .- 1)

        # Initialize arrays

        vel = zeros(ns...)
        fact = zeros(ns...)
        snapshots = (snapevery !== nothing ? zeros(ns..., div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients(halo) for _ in 1:N]...)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimul{N}, parall)

        return new(
            ls,
            ns,
            gridspacing,
            nt,
            dt,
            halo,
            rcoef,
            freetop,
            snapevery,
            snapshots,
            infoevery,
            vel,
            fact,
            cpmlcoeffs,
            backend
        )
    end
end

# Traits for AcousticCDCPMLWaveSimul
IsSnappableTrait(::Type{<:AcousticCDCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticCDCPMLWaveSimul}) = LocalGrid()

# Backend selections for AcousticCDCPMLWaveSimul
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:serial}}) =
    Acoustic1D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:serial}}) =
    Acoustic2D_CD_CPML_Serial
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:serial}}) =
    Acoustic3D_CD_CPML_Serial

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:threads}}) =
    Acoustic1D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:threads}}) =
    Acoustic2D_CD_CPML_Threads
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:threads}}) =
    Acoustic3D_CD_CPML_Threads

select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{1}}, ::Type{Val{:GPU}}) =
    Acoustic1D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{2}}, ::Type{Val{:GPU}}) =
    Acoustic2D_CD_CPML_GPU
select_backend(::CPMLBoundaryCondition, ::LocalGrid, ::Type{<:AcousticCDCPMLWaveSimul{3}}, ::Type{Val{:GPU}}) =
    Acoustic3D_CD_CPML_GPU

#######################################################################

struct AcousticCDReflWaveSimul{N} <: AcousticCDWaveSimul{N} end    # TODO implementation

IsSnappableTrait(::Type{<:AcousticCDReflWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDReflWaveSimul}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:AcousticCDReflWaveSimul}) = LocalGrid()
