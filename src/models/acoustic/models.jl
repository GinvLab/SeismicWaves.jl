struct AcousticCPMLWaveModel{N} <: WaveModel{N}
    nt::Integer
    ns::NTuple{N, <:Integer}
    ls::NTuple{N, <:Integer}
    dt::Real
    Δs::NTuple{N, <:Real}
    vel::Array{<:Real, N}
    fact::Array{<:Real, N}
    halo::Integer
    rcoef::Real
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    freetop::Bool
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{<:Array{<:Real}, Nothing}
    infoevery::Integer

    function AcousticCPMLWaveModel{N}(
        nt::Integer,
        dt::Real,
        Δs::NTuple{N, <:Real},
        halo::Integer,
        rcoef::Real,
        vel::Array{<:Real, N};
        freetop::Bool = true,
        snapevery::Union{<:Integer, Nothing} = nothing,
        infoevery::Union{<:Integer, Nothing} = nothing
    ) where {N}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert all(d -> d > 0, Δs) "All cell sizes must be positive!"
        
        # Check velocity
        ns = size(vel)
        ns_cpml = freetop ? ns[1:end-1] : ns
        @assert all(n -> n >= 2halo+3, ns_cpml) "Velocity model in the dimensions with C-PML boundaries must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model sizes
        ls = Δs .* (ns .- 1)

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

        new(nt, ns, ls, dt, Δs, vel, fact, halo, rcoef, cpmlcoeffs, freetop, snapevery, snapshots, infoevery)
    end
end


AcousticCPMLWaveModel1D = AcousticCPMLWaveModel{1}
WaveEquationTrait(::Type{<:AcousticCPMLWaveModel1D}) = AcousticWaveEquation()
BoundaryConditionTrait(::Type{<:AcousticCPMLWaveModel1D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:AcousticCPMLWaveModel1D}) = Snappable()
GridTrait(::Type{<:AcousticCPMLWaveModel1D}) = LocalGrid()

AcousticCPMLWaveModel2D = AcousticCPMLWaveModel{2}
WaveEquationTrait(::Type{<:AcousticCPMLWaveModel2D}) = AcousticWaveEquation()
BoundaryConditionTrait(::Type{<:AcousticCPMLWaveModel2D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:AcousticCPMLWaveModel2D}) = Snappable()
GridTrait(::Type{<:AcousticCPMLWaveModel2D}) = LocalGrid()

AcousticCPMLWaveModel3D = AcousticCPMLWaveModel{3}
WaveEquationTrait(::Type{<:AcousticCPMLWaveModel3D}) = AcousticWaveEquation()
BoundaryConditionTrait(::Type{<:AcousticCPMLWaveModel3D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:AcousticCPMLWaveModel3D}) = Snappable()
GridTrait(::Type{<:AcousticCPMLWaveModel3D}) = LocalGrid()
