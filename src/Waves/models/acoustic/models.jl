"""
Simplest model for isotropic acoustic 1D wave equation modeling with reflective boundaries.
"""
struct IsotropicAcousticSerialReflectiveWaveModel1D{T<:Real} <: WaveModel1D
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
        IsotropicAcousticSerialReflectiveWaveModel1D[{T<:Real = Float64}](
            nt::Integer,
            dt::Real,
            dx::Real,
            vel::Vector{T},
            snapevery::Union{Integer, Nothing} = nothing
        )
    
    Construct an isotropic acoustic 1D wave equation model with reflective boundaries and serial kernel.

    # Arguments
    - `nt::Integer`: the number of timesteps.
    - `dt::Real`: the timestep size.
    - `dx::Real`: the size of a grid cell in x-dimension.
    - `vel::Vector{T}`: the velocity model.
    - `snapevery::Union{Integer, Nothing} = nothing`: if specified, saves pressure field every `snapevery` timesteps into the `snapshots` field.
    """
    function IsotropicAcousticSerialReflectiveWaveModel1D{T}(
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
IsotropicAcousticSerialReflectiveWaveModel1D(nt, dt, dx, vel, snapevery=nothing) = IsotropicAcousticSerialReflectiveWaveModel1D{Float64}(nt, dt, dx, vel, snapevery)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel1D}) = IsotropicAcousticWaveEquation()
KernelTypeTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel1D}) = SerialKernel()
BoundaryConditionTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel1D}) = ReflectiveBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel1D}) = Snappable()
