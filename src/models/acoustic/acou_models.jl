
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

    function AcousticCDCPMLWaveSimul{N}(
        ns::NTuple{N, <:Integer},
        gridspacing::NTuple{N, <:Real},
        nt::Integer,
        dt::Real,
        halo::Integer,
        rcoef::Real;
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

        return new(ls, ns, gridspacing, nt, dt, halo, rcoef, freetop, snapevery, snapshots, infoevery, vel, fact, cpmlcoeffs)
    end
end

IsSnappableTrait(::Type{<:AcousticCDCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticCDCPMLWaveSimul}) = LocalGrid()

#######################################################################

struct AcousticCDReflWaveSimul{N} <: AcousticCDWaveSimul{N} end    # TODO implementation

IsSnappableTrait(::Type{<:AcousticCDReflWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDReflWaveSimul}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:AcousticCDReflWaveSimul}) = LocalGrid()

#######################################################################
