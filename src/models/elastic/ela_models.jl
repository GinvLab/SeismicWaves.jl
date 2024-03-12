###########################################################

# Functions for all ElasticIsoWaveSimul subtypes

@views function check_matprop(wavsim::ElasticIsoWaveSimul{N}, matprop::ElasticIsoMaterialProperties{N}) where {N}
    # Checks
    vp = sqrt.((matprop.λ .+ 2.0 * matprop.μ) ./ matprop.ρ)
    @assert ndims(vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(vp) == wavsim.gridsize "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(wavsim.gridsize)"
    @assert all(matprop.λ .> 0) "Lamè coefficient λ must be positive!"
    @assert all(matprop.μ .> 0) "Lamè coefficient μ must be positive!"
    @assert all(matprop.ρ .> 0) "Density must be positive!"

    # Check courant condition
    vel_max = get_maximum_func(wavsim)(vp)
    tmp = sqrt.(sum(1 ./ wavsim.gridspacing .^ 2))
    courant = vel_max * wavsim.dt * tmp
    @debug "Courant number: $(courant)"
    if courant > 1.0
        @warn "Courant condition not satisfied! [$(courant)]"
    end

    return
end

function check_numerics(
    wavsim::ElasticIsoWaveSimul,
    shot::Shot;
    min_ppw::Integer=10
)
    # Check points per wavelengh
    # min Vs
    vel_min = get_minimum_func(wavsim)(sqrt.(wavsim.matprop.μ ./ wavsim.matprop.ρ))
    h_max = maximum(wavsim.gridspacing)
    fmax = shot.srcs.domfreq * 2.0
    ppw = vel_min / (fmax * h_max)
    @debug "Points per wavelength: $(ppw)"

    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    return
end

@views function update_matprop!(wavsim::ElasticIsoWaveSimul{N}, matprop::ElasticIsoMaterialProperties{N}) where {N}

    # Update material properties
    wavsim.matprop.λ .= matprop.λ
    wavsim.matprop.μ .= matprop.μ
    wavsim.matprop.ρ .= matprop.ρ

    # the following on device?
    precomp_elaprop!(wavsim.matprop)

    return
end

###########################################################

struct Elasticψdomain2D{T <: AbstractFloat}
    ψ_∂σxx∂x::Array{T, 2}
    ψ_∂σxz∂z::Array{T, 2}
    ψ_∂σxz∂x::Array{T, 2}
    ψ_∂σzz∂z::Array{T, 2}
    ψ_∂vx∂x::Array{T, 2}
    ψ_∂vz∂z::Array{T, 2}
    ψ_∂vz∂x::Array{T, 2}
    ψ_∂vx∂z::Array{T, 2}

    function Elasticψdomain2D(backend, gridsize, halo)
        @assert length(gridsize) == 2
        ψ_gridsize = [gridsize...]

        gs1, gs2 = copy(ψ_gridsize), copy(ψ_gridsize)
        gs1[1] = 2 * halo
        gs2[2] = 2 * halo

        ψ_∂σxx∂x = backend.zeros(gs1...)
        ψ_∂σxz∂z = backend.zeros(gs2...)
        ψ_∂σxz∂x = backend.zeros(gs1...)
        ψ_∂σzz∂z = backend.zeros(gs2...)
        ψ_∂vx∂x = backend.zeros(gs1...)
        ψ_∂vz∂z = backend.zeros(gs2...)
        ψ_∂vz∂x = backend.zeros(gs1...)
        ψ_∂vx∂z = backend.zeros(gs2...)

        T = eltype(ψ_∂σxx∂x)
        return new{T}(ψ_∂σxx∂x,
            ψ_∂σxz∂z,
            ψ_∂σxz∂x,
            ψ_∂σzz∂z,
            ψ_∂vx∂x,
            ψ_∂vz∂z,
            ψ_∂vz∂x,
            ψ_∂vx∂z)
    end
end

struct Velpartic2D{T <: AbstractFloat}
    vx::Array{T, 2}
    vz::Array{T, 2}
end

struct Stress2D{T <: AbstractFloat}
    σxx::Array{T, 2}
    σzz::Array{T, 2}
    σxz::Array{T, 2}
end

##############################################################

struct ElasticIsoCPMLWaveSimul{N} <: ElasticIsoWaveSimul{N}
    # Physics
    domainextent::NTuple{N, <:Real}
    # Numerics
    gridsize::NTuple{N, <:Integer}
    gridspacing::NTuple{N, <:Real}
    nt::Integer
    dt::Real
    # BDC and CPML parameters
    halo::Integer
    rcoef::Real
    freetop::Bool
    # Gradient computation setup
    gradient::Bool
    check_freq::Union{<:Integer, Nothing}
    # Snapshots
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{Vector{<:Array{<:Real}}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Material properties
    matprop::AbstrElasticIsoMaterialProperties
    # Forward computation arrays
    velpartic::Any # 2D: 2 comp, 3D: 3 comp
    stress::Any # 2D: 3 arrays, 3D: 6 arrays
    # CPML coefficients
    cpmlcoeffs::Any
    ψ::Any
    # Gradient computation arrays
    adj::Any
    ψ_adj::Any
    grad::Any
    # Checkpointing setup
    last_checkpoint::Union{<:Integer, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_ψ::Any
    # Backend
    backend::Module
    parall::Symbol

    function ElasticIsoCPMLWaveSimul{N}(
        gridsize::NTuple{N, <:Integer},
        gridspacing::NTuple{N, <:Real},
        nt::Integer,
        dt::Real,
        halo::Integer,
        rcoef::Real;
        parall::Symbol=:serial,
        freetop::Bool=true,
        gradient::Bool=false,
        check_freq::Union{<:Integer, Nothing}=nothing,
        snapevery::Union{<:Integer, Nothing}=nothing,
        infoevery::Union{<:Integer, Nothing}=nothing
    ) where {N}
        # Check numerics
        @assert all(gridsize .> 0) "All numbers of grid points must be positive!"
        @assert all(gridspacing .> 0) "All grid spacings must be positive!"
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"

        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        gridsize_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, gridsize_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Compute wavsim sizes
        domainextent = gridspacing .* (gridsize .- 1)

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimul{N}, parall)

        # Initialize material properties
        if N == 2
            matprop = ElasticIsoMaterialProperties2D(; λ=backend.zeros(gridsize...),
                μ=backend.zeros(gridsize...),
                ρ=backend.zeros(gridsize...),
                λ_ihalf=backend.zeros((gridsize .- [1, 0])...),
                μ_ihalf=backend.zeros((gridsize .- [1, 0])...),
                μ_jhalf=backend.zeros((gridsize .- [0, 1])...),
                ρ_ihalf_jhalf=backend.zeros((gridsize .- 1)...)
            )

        else
            error("Only elastic 2D is currently implemented.")
        end

        # Initialize computational arrays
        if N == 2
            velpartic = Velpartic2D([backend.zeros(gridsize...) for _ in 1:N]...)  # vx, vy[, vz]
            stress = Stress2D([backend.zeros(gridsize...) for _ in 1:(N-1)*3]...)  # vx, vy[, vz]
        end

        ##
        if N == 2 # 2D
            ψ = Elasticψdomain2D(backend, gridsize, halo)
        else
            error("Only elastic 2D is currently implemented.")
        end

        # Initialize CPML coefficients
        cpmlcoeffs = [CPMLCoefficientsAxis(halo, backend) for _ in 1:N]

        # Initialize gradient arrays if needed
        if gradient
            error("Gradient for elastic calculations not yet implemented!")

            # # Current gradient array
            # curgrad = backend.zeros(gridsize...)
            # # Adjoint arrays
            # adj = backend.zeros(gridsize...)
            # # Initialize CPML arrays
            # ψ_adj = []
            # ξ_adj = []
            # for i in 1:N
            #     ψ_gridsize = [gridsize...]
            #     ξ_gridsize = [gridsize...]
            #     ψ_gridsize[i] = halo + 1
            #     ξ_gridsize[i] = halo
            #     append!(ψ_adj, [backend.zeros(ψ_gridsize...), backend.zeros(ψ_gridsize...)])
            #     append!(ξ_adj, [backend.zeros(ξ_gridsize...), backend.zeros(ξ_gridsize...)])
            # end
            # # Checkpointing setup
            # if check_freq !== nothing
            #     @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
            #     @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
            #     # Time step of last checkpoint
            #     last_checkpoint = floor(Int, nt / check_freq) * check_freq
            #     # Checkpointing arrays
            #     save_buffer = backend.zeros(gridsize..., check_freq + 2)      # pressure window buffer
            #     checkpoints = Dict{Int, backend.Data.Array}()           # pressure checkpoints
            #     checkpoints_ψ = Dict{Int, Any}()                        # ψ arrays checkpoints
            #     checkpoints_ξ = Dict{Int, Any}()                        # ξ arrays checkpoints
            #     # Save initial conditions as first checkpoint
            #     checkpoints[-1] = copy(pold)
            #     checkpoints[0] = copy(pcur)
            #     checkpoints_ψ[0] = copy.(ψ)
            #     checkpoints_ξ[0] = copy.(ξ)
            #     # Preallocate future checkpoints
            #     for it in 1:(nt+1)
            #         if it % check_freq == 0
            #             checkpoints[it] = backend.zeros(gridsize...)
            #             checkpoints[it-1] = backend.zeros(gridsize...)
            #             checkpoints_ψ[it] = copy.(ψ)
            #             checkpoints_ξ[it] = copy.(ξ)
            #         end
            #     end
            # else    # no checkpointing
            #     last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
            #     save_buffer = backend.zeros(gridsize..., nt + 2)          # save all timesteps (from -1 to nt+1 so nt+2)
            #     checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
            #     checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
            #     checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            # end
            # # Save first 2 timesteps in save buffer
            # save_buffer[fill(Colon(), N)..., 1] .= pold
            # save_buffer[fill(Colon(), N)..., 2] .= pcur
        end

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? [zeros(gridsize..., div(nt, snapevery)) for _ in 1:N] : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        return new(
            domainextent,
            gridsize,
            gridspacing,
            nt,
            dt,
            halo,
            rcoef,
            freetop,
            gradient,
            gradient ? check_freq : nothing,
            snapevery,
            snapshots,
            infoevery,
            matprop,
            velpartic,
            stress,
            cpmlcoeffs,
            ψ,
            gradient ? adj : nothing,
            gradient ? ψ_adj : nothing,
            gradient ? grad : nothing,
            gradient ? last_checkpoint : nothing,
            gradient ? save_buffer : nothing,
            gradient ? checkpoints : nothing,
            gradient ? checkpoints_ψ : nothing,
            backend,
            parall
        )
    end
end

###########################################################

# Specific functions for ElasticIsoCPMLWaveSimul

@views function reset!(wavsim::ElasticIsoCPMLWaveSimul{N}) where {N}

    # Reset computational arrays
    for p in propertynames(wavsim.velpartic)
        getfield(wavsim.velpartic, p) .= 0.0
    end
    for p in propertynames(wavsim.stress)
        getfield(wavsim.stress, p) .= 0.0
    end
    for p in propertynames(wavsim.ψ)
        getfield(wavsim.ψ, p) .= 0.0
    end

    # Reset gradient arrays
    if wavsim.gradient
        for p in eachindex(wavsim.adj)
            wavsim.adj[p][:] .= 0.0
        end
        for p in eachindex(wavsim.grad)
            wavsim.grad[p] .= 0.0
        end
        for p in propertynames(wavsim.ψ_adj)
            getfield(wavsim.ψ_adj, p) .= 0.0
        end
    end
end
###########################################################

# Traits for ElasticIsoCPMLWaveSimul

IsSnappableTrait(::Type{<:ElasticIsoCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:ElasticIsoCPMLWaveSimul}) = LocalGrid()

###########################################################

struct ElasticIsoReflWaveSimul{N} <: ElasticIsoWaveSimul{N} end    # TODO implementation

###########################################################

# Traits for ElasticIsoReflWaveSimul

IsSnappableTrait(::Type{<:ElasticIsoReflWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoReflWaveSimul}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:ElasticIsoReflWaveSimul}) = LocalGrid()

###########################################################
