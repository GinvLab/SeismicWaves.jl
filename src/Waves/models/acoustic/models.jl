"""
Simplest model for isotropic acoustic 2D wave equation modeling with reflective boundaries.
"""
struct IsotropicAcousticSerialReflectiveWaveModel2D{T<:Real} <: WaveModel2D
    nt::Integer
    nx::Integer
    nz::Integer
    dt::Real
    dx::Real
    dz::Real
    vel::Matrix{T}
    fact_x::Matrix{T}
    fact_z::Matrix{T}
    pold::Matrix{T}
    pcur::Matrix{T}
    pnew::Matrix{T}
    snapevery::Union{Integer, Nothing}
    snapshots::Union{Array{T, 3}, Nothing}

    @doc """
        IsotropicAcousticSerialReflectiveWaveModel2D[{T<:Real = Float64}](
            nt::Integer,
            dt::Real,
            dx::Real,
            dz::Real,
            vel::Matrix{T},
            savesnapshots::Bool = true
        )
    
    Construct an isotropic acoustic 2D wave equation model with reflective boundaries and serial kernel.

    # Arguments
    - `nt::Integer`: the number of timesteps.
    - `dt::Real`: the timestep size.
    - `dx::Real`: the size of a grid cell in x-dimension.
    - `dz::Real`: the size fo a grid cell in z-dimension.
    - `vel::Matrix{T}`: the velocity model.
    - `snapevery::Union{Integer, Nothing} = nothing`: if specified, saves pressure field every `snapevery` timesteps into the `snapshots` field.
    """
    function IsotropicAcousticSerialReflectiveWaveModel2D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        dz::Real,
        vel::Matrix{T},
        snapevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        @assert dz > 0 "Cell size in z-direction must be positive!"
        
        # Check velocity
        nx, nz = size(vel)
        @assert nx >= 3 "Velocity model in x-direction must have at least 3 cells!"
        @assert nz >= 3 "Velocity model in z-direction must have at least 3 cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Initialize arrays
        fact_x = zero(vel)
        fact_z = zero(vel)
        pold = zero(vel)
        pcur = zero(vel)
        pnew = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, nz, div(nt, snapevery)) : nothing)

        new(nt, nx, nz, dt, dx, dz, vel, fact_x, fact_z, pold, pcur, pnew, snapevery, snapshots)
    end
end

# Default type parameter constuctor
IsotropicAcousticSerialReflectiveWaveModel2D(nt, dt, dx, dz, vel, snapevery=nothing) = IsotropicAcousticSerialReflectiveWaveModel2D{Float64}(nt, dt, dx, dz, vel, snapevery)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel2D}) = IsotropicAcoustic()
KernelTypeTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel2D}) = SerialKernel()
BoundaryConditionTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel2D}) = Reflective()
IsSnappableTrait(::Type{<:IsotropicAcousticSerialReflectiveWaveModel2D}) = Snappable()
