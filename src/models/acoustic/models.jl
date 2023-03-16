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
    infoevery::Integer

    @doc """
    Construct an isotropic acoustic 1D wave equation model with reflective boundaries.
    """
    function IsotropicAcousticReflectiveWaveModel1D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        vel::Vector{T};
        snapevery::Union{Integer, Nothing} = nothing,
        infoevery::Union{Integer, Nothing} = nothing
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

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new(nt, nx, lx, dt, dx, vel, fact, snapevery, snapshots, infoevery)
    end
end

# Default type parameter constuctor
IsotropicAcousticReflectiveWaveModel1D(args...; kwargs...) = IsotropicAcousticReflectiveWaveModel1D{Float64}(args...; kwargs...)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = ReflectiveBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = Snappable()
GridTrait(::Type{<:IsotropicAcousticReflectiveWaveModel1D}) = LocalGrid()

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
    infoevery::Integer

    @doc """
    Construct an isotropic acoustic 1D wave equation model with CPML boundaries.
    """
    function IsotropicAcousticCPMLWaveModel1D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        halo::Integer,
        rcoef::Real,
        vel::Vector{T};
        snapevery::Union{Integer, Nothing} = nothing,
        infoevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        
        # Check velocity
        nx = length(vel)
        @assert nx >= 2halo+3 "Velocity model in x-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model size
        lx = dx * (nx-1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs = CPMLCoefficients(halo)

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new(nt, nx, lx, dt, dx, vel, fact, halo, rcoef, cpmlcoeffs, snapevery, snapshots, infoevery)
    end
end

# Default type parameter constuctor
IsotropicAcousticCPMLWaveModel1D(args...; kwargs...) = IsotropicAcousticCPMLWaveModel1D{Float64}(args...; kwargs...)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = Snappable()
GridTrait(::Type{<:IsotropicAcousticCPMLWaveModel1D}) = LocalGrid()


"""
Isotropic acoustic 2D wave equation model with CPML boundaries.
"""
struct IsotropicAcousticCPMLWaveModel2D{T<:Real} <: WaveModel2D
    nt::Integer
    nx::Integer
    ny::Integer
    lx::Real
    ly::Real
    dt::Real
    dx::Real
    dy::Real
    vel::Matrix{T}
    fact::Matrix{T}
    halo::Integer
    rcoef::Real
    cpmlcoeffs_x::CPMLCoefficients
    cpmlcoeffs_y::CPMLCoefficients
    freetop::Bool
    snapevery::Union{Integer, Nothing}
    snapshots::Union{Array{T, 3}, Nothing}
    infoevery::Integer

    @doc """
    Construct an isotropic acoustic 2D wave equation model with CPML boundaries.
    """
    function IsotropicAcousticCPMLWaveModel2D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        dy::Real,
        halo::Integer,
        rcoef::Real,
        vel::Matrix{T};
        freetop::Bool = true,
        snapevery::Union{Integer, Nothing} = nothing,
        infoevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        @assert dy > 0 "Cell size in y-direction must be positive!"
        
        # Check velocity
        nx, ny = size(vel)
        @assert nx >= 2halo+3 "Velocity model in x-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert ny >= 2halo+3 "Velocity model in y-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model size
        lx = dx * (nx-1)
        ly = dy * (ny-1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, ny, div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs_x = CPMLCoefficients(halo)
        cpmlcoeffs_y = CPMLCoefficients(halo)

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new(nt, nx, ny, lx, ly, dt, dx, dy, vel, fact, halo, rcoef, cpmlcoeffs_x, cpmlcoeffs_y, freetop, snapevery, snapshots, infoevery)
    end
end

# Default type parameter constuctor
IsotropicAcousticCPMLWaveModel2D(args...; kwargs...) = IsotropicAcousticCPMLWaveModel2D{Float64}(args...; kwargs...)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticCPMLWaveModel2D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticCPMLWaveModel2D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticCPMLWaveModel2D}) = Snappable()
GridTrait(::Type{<:IsotropicAcousticCPMLWaveModel2D}) = LocalGrid()

"""
Isotropic acoustic 3D wave equation model with CPML boundaries.
"""
struct IsotropicAcousticCPMLWaveModel3D{T<:Real} <: WaveModel3D
    nt::Integer
    nx::Integer
    ny::Integer
    nz::Integer
    lx::Real
    ly::Real
    lz::Real
    dt::Real
    dx::Real
    dy::Real
    dz::Real
    vel::Array{T, 3}
    fact::Array{T, 3}
    halo::Integer
    rcoef::Real
    cpmlcoeffs_x::CPMLCoefficients
    cpmlcoeffs_y::CPMLCoefficients
    cpmlcoeffs_z::CPMLCoefficients
    freetop::Bool
    snapevery::Union{Integer, Nothing}
    snapshots::Union{Array{T, 4}, Nothing}
    infoevery::Integer

    @doc """
    Construct an isotropic acoustic 3D wave equation model with CPML boundaries.
    """
    function IsotropicAcousticCPMLWaveModel3D{T}(
        nt::Integer,
        dt::Real,
        dx::Real,
        dy::Real,
        dz::Real,
        halo::Integer,
        rcoef::Real,
        vel::Array{T, 3};
        freetop::Bool = true,
        snapevery::Union{Integer, Nothing} = nothing,
        infoevery::Union{Integer, Nothing} = nothing
    ) where {T<:Real}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"
        @assert dx > 0 "Cell size in x-direction must be positive!"
        @assert dy > 0 "Cell size in y-direction must be positive!"
        @assert dz > 0 "Cell size in z-direction must be positive!"
        
        # Check velocity
        nx, ny, nz = size(vel)
        @assert nx >= 2halo+3 "Velocity model in x-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert ny >= 2halo+3 "Velocity model in y-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert nz >= 2halo+3 "Velocity model in z-direction must have at least 2*halo+3 = $(2halo+3) cells!"
        @assert all(vel .> 0.0) "Velocity model must be positive everywhere!"

        # Compute model size
        lx = dx * (nx-1)
        ly = dy * (ny-1)
        lz = dz * (nz-1)

        # Initialize arrays
        fact = zero(vel)
        snapshots = (snapevery !== nothing ? zeros(nx, ny, nz, div(nt, snapevery)) : nothing)
        # Create CPML coefficients
        cpmlcoeffs_x = CPMLCoefficients(halo)
        cpmlcoeffs_y = CPMLCoefficients(halo)
        cpmlcoeffs_z = CPMLCoefficients(halo)

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new(nt, nx, ny, nz, lx, ly, lz, dt, dx, dy, dz, vel, fact, halo, rcoef, cpmlcoeffs_x, cpmlcoeffs_y, cpmlcoeffs_z, freetop, snapevery, snapshots, infoevery)
    end
end

# Default type parameter constuctor
IsotropicAcousticCPMLWaveModel3D(args...; kwargs...) = IsotropicAcousticCPMLWaveModel3D{Float64}(args...; kwargs...)

# Tag traits
WaveEquationTrait(::Type{<:IsotropicAcousticCPMLWaveModel3D}) = IsotropicAcousticWaveEquation()
BoundaryConditionTrait(::Type{<:IsotropicAcousticCPMLWaveModel3D}) = CPMLBoundaryCondition()
IsSnappableTrait(::Type{<:IsotropicAcousticCPMLWaveModel3D}) = Snappable()
GridTrait(::Type{<:IsotropicAcousticCPMLWaveModel3D}) = LocalGrid()
