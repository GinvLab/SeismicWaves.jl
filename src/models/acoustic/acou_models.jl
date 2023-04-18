
###########################################################

struct AcousticCDCPMLWaveSimul{N} <: AcousticCDWaveSimul{N}
    nt::Integer
    ns::NTuple{N, <:Integer}
    ls::NTuple{N, <:Integer}
    dt::Real
    gridspacing::NTuple{N, <:Real}
    vel::Array{<:Real, N}
    fact::Array{<:Real, N}
    halo::Integer
    rcoef::Real
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    freetop::Bool
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{<:Array{<:Real}, Nothing}
    infoevery::Integer

    function AcousticCDCPMLWaveSimul{N}(
        nt::Integer,
        dt::Real,
        gridspacing::NTuple{N, <:Real},
        halo::Integer,
        rcoef::Real,
        vel::Array{<:Real, N};
        freetop::Bool=true,
        snapevery::Union{<:Integer, Nothing}=nothing,
        infoevery::Union{<:Integer, Nothing}=nothing
    ) where {N}

        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert all(d -> d > 0, gridspacing) "All cell sizes must be positive!"

        # Check velocity
        ns = size(vel)
        ns_cpml = freetop ? ns[1:(end-1)] : ns
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Velocity model in the dimensions with C-PML boundaries must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model sizes
        ls = gridspacing .* (ns .- 1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(ns..., div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients(halo) for _ in 1:N]...)

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        return new(nt, ns, ls, dt, gridspacing, vel, fact, halo, rcoef, cpmlcoeffs, freetop, snapevery, snapshots, infoevery)
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
