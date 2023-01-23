"""
Simplest model for isotropic acoustic 1D wave equation modeling with reflective boundaries.
"""
struct IsotropicAcousticReflectiveWaveModel1D{T<:Real} <: WaveModel1D
    nt::Integer
    nx::Integer
    lx::Real
    dt::Real
    dx::Real
    vel::Vector{T}
    fact::Vector{T}
    snapevery::Union{Integer, Nothing}
    snapshots::Union{Matrix{T}, Nothing}

    @doc """
        IsotropicAcousticReflectiveWaveModel1D[{T<:Real = Float64}](
            nt::Integer,
            dt::Real,
            dx::Real,
            vel::Vector{T},
            snapevery::Union{Integer, Nothing} = nothing
        )
    
    Construct an isotropic acoustic 1D wave equation model with reflective boundaries.

    # Arguments
    - `nt::Integer`: the number of timesteps.
    - `dt::Real`: the timestep size.
    - `dx::Real`: the size of a grid cell in x-dimension.
    - `vel::Vector{T}`: the velocity model.
    - `snapevery::Union{Integer, Nothing} = nothing`: if specified, saves pressure field every `snapevery` timesteps into the `snapshots` field.
    """
    function IsotropicAcousticReflectiveWaveModel1D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        vel::Vector{T},
        snapevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        
        # Check velocity
        nx = length(vel)
        @assert nx >= 3 "Velocity model in x-direction must have at least 3 cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model size
        lx = dx * (nx-1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, div(nt, snapevery)) : nothing)

        new(nt, nx, lx, dt, dx, vel, fact, snapevery, snapshots)
    end
end

# Default type parameter constuctor
IsotropicAcousticReflectiveWaveModel1D(nt, dt, dx, vel, snapevery=nothing) = IsotropicAcousticReflectiveWaveModel1D{Float64}(nt, dt, dx, vel, snapevery)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = ReflectiveBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = Snappable()

"""
Isotropic acoustic 1D wave equation model with CPML boundaries.
"""
struct IsotropicAcousticCPMLWaveModel1D{T<:Real} <: WaveModel1D
    nt::Integer
    nx::Integer
    lx::Real
    dt::Real
    dx::Real
    vel::Vector{T}
    fact::Vector{T}
    halo::Integer
    rcoef::Real
    cpmlcoeffs::CPMLCoefficients
    snapevery::Union{Integer, Nothing}
    snapshots::Union{Matrix{T}, Nothing}

    @doc """
        IsotropicAcousticCPMLWaveModel1D[{T<:Real = Float64}](
            nt::Integer,
            dt::Real,
            dx::Real,
            halo::Integer,
            rcoef::Real,
            vel::Vector{T},
            snapevery::Union{Integer, Nothing} = nothing
        )
    
    Construct an isotropic acoustic 1D wave equation model with CPML boundaries.

    # Arguments
    - `nt::Integer`: the number of timesteps.
    - `dt::Real`: the timestep size.
    - `dx::Real`: the size of a grid cell in x-dimension.
    - `vel::Vector{T}`: the velocity model.
    - `snapevery::Union{Integer, Nothing} = nothing`: if specified, saves pressure field every `snapevery` timesteps into the `snapshots` field.
    """
    function IsotropicAcousticCPMLWaveModel1D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        halo::Integer,
        rcoef::Real,
        vel::Vector{T},
        snapevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        
        # Check velocity
        nx = length(vel)
        @assert nx >= 3 "Velocity model in x-direction must have at least 3 cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model size
        lx = dx * (nx-1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs = CPMLCoefficients(halo)

        new(nt, nx, lx, dt, dx, vel, fact, halo, rcoef, cpmlcoeffs, snapevery, snapshots)
    end
end

# Default type parameter constuctor
IsotropicAcousticCPMLWaveModel1D(nt, dt, dx, halo, rcoef, vel, snapevery=nothing) = IsotropicAcousticCPMLWaveModel1D{Float64}(nt, dt, dx, halo, rcoef, vel, snapevery)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = Snappable()
