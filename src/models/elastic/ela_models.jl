###########################################################

# Functions for all ElasticIsoWaveSimulation subtypes

@views function check_matprop(model::ElasticIsoWaveSimulation{T, N}, matprop::ElasticIsoMaterialProperties{T, N}) where {T, N}
    # Checks
    vp = sqrt.((matprop.λ .+ 2.0 * matprop.μ) ./ matprop.ρ)
    @assert ndims(vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(vp) == model.gridsize "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.gridsize)"
    @assert all(matprop.λ .> 0) "Lamè coefficient λ must be positive!"
    @assert all(matprop.μ .> 0) "Lamè coefficient μ must be positive!"
    @assert all(matprop.ρ .> 0) "Density must be positive!"

    # Check courant condition
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt.(sum(1 ./ model.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp
    @info "Courant number: $(courant)"
    if courant > 1.0
        @warn "Courant condition not satisfied! [$(courant)]"
    end

    return
end

function check_numerics(
    model::ElasticIsoWaveSimulation,
    shot::Shot;
    min_ppw::Int=10
)
    # Check points per wavelengh
    # min Vs
    vel_min = get_minimum_func(model)(sqrt.(model.matprop.μ ./ model.matprop.ρ))
    h_max = maximum(model.gridspacing)
    fmax = shot.srcs.domfreq * 2.0
    ppw = vel_min / (fmax * h_max)
    @info "Points per wavelength: $(ppw)"

    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    return
end

@views function update_matprop!(model::ElasticIsoWaveSimulation{T, N}, matprop::ElasticIsoMaterialProperties{T, N}) where {T, N}

    # Update material properties
    model.matprop.λ .= matprop.λ
    model.matprop.μ .= matprop.μ
    model.matprop.ρ .= matprop.ρ

    # the following on device?
    precomp_elaprop!(model.matprop)

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

        ψ_∂σxx∂x = backend.zeros(T, gs1...)
        ψ_∂σxz∂z = backend.zeros(T, gs2...)
        ψ_∂σxz∂x = backend.zeros(T, gs1...)
        ψ_∂σzz∂z = backend.zeros(T, gs2...)
        ψ_∂vx∂x = backend.zeros(T, gs1...)
        ψ_∂vz∂z = backend.zeros(T, gs2...)
        ψ_∂vz∂x = backend.zeros(T, gs1...)
        ψ_∂vx∂z = backend.zeros(T, gs2...)

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

struct ElasticIsoCPMLWaveSimulation{T, N} <: ElasticIsoWaveSimulation{T, N}
    # Physics
    domainextent::NTuple{N, T}
    # Numerics
    gridsize::NTuple{N, Int}
    gridspacing::NTuple{N, T}
    nt::Int
    dt::T
    # BDC and CPML parameters
    halo::Int
    rcoef::T
    freetop::Bool
    # Gradient computation setup
    gradient::Bool
    check_freq::Union{Int, Nothing}
    # Snapshots
    snapevery::Union{Int, Nothing}
    snapshots::Union{Vector{<:Array{T}}, Nothing}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::AbstrElasticIsoMaterialProperties{T, N}
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
    last_checkpoint::Union{Int, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_ψ::Any
    # Backend
    backend::Module
    parall::Symbol

    function ElasticIsoCPMLWaveSimulation(
        gridsize::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        nt::Int,
        dt::T,
        matprop::ElasticIsoMaterialProperties{T, N},
        halo::Int,
        rcoef::T;
        parall::Symbol=:serial,
        freetop::Bool=true,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing
    ) where {T, N}
        # Check numerics
        @assert all(gridsize .> 0) "All numbers of grid points must be positive!"
        @assert all(gridspacing .> 0) "All grid spacings must be positive!"
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"

        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        gridsize_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, gridsize_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Compute model sizes
        domainextent = gridspacing .* (gridsize .- 1)

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimulation{T, N}, parall)
        V = backend.Data.Array{T, 1}

        # Initialize computational arrays
        if N == 2
            velpartic = Velpartic2D([backend.zeros(T, gridsize...) for _ in 1:N]...)  # vx, vy[, vz]
            stress = Stress2D([backend.zeros(T, gridsize...) for _ in 1:(N-1)*3]...)  # vx, vy[, vz]
        end

        ##
        if N == 2 # 2D
            ψ = Elasticψdomain2D(backend, gridsize, halo)
        else
            error("Only elastic 2D is currently implemented.")
        end

        # Initialize CPML coefficients
        cpmlcoeffs = [CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]

        # Initialize gradient arrays if needed
        if gradient
            error("Gradient for elastic calculations not yet implemented!")

            # # Current gradient array
            # curgrad = backend.zeros(T, gridsize...)
            # # Adjoint arrays
            # adj = backend.zeros(T, gridsize...)
            # # Initialize CPML arrays
            # ψ_adj = []
            # ξ_adj = []
            # for i in 1:N
            #     ψ_gridsize = [gridsize...]
            #     ξ_gridsize = [gridsize...]
            #     ψ_gridsize[i] = halo + 1
            #     ξ_gridsize[i] = halo
            #     append!(ψ_adj, [backend.zeros(T, ψ_gridsize...), backend.zeros(T, ψ_gridsize...)])
            #     append!(ξ_adj, [backend.zeros(T, ξ_gridsize...), backend.zeros(T, ξ_gridsize...)])
            # end
            # # Checkpointing setup
            # if check_freq !== nothing
            #     @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
            #     @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
            #     # Time step of last checkpoint
            #     last_checkpoint = floor(Int, nt / check_freq) * check_freq
            #     # Checkpointing arrays
            #     save_buffer = backend.zeros(T, gridsize..., check_freq + 2)      # pressure window buffer
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
            #             checkpoints[it] = backend.zeros(T, gridsize...)
            #             checkpoints[it-1] = backend.zeros(T, gridsize...)
            #             checkpoints_ψ[it] = copy.(ψ)
            #             checkpoints_ξ[it] = copy.(ξ)
            #         end
            #     end
            # else    # no checkpointing
            #     last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
            #     save_buffer = backend.zeros(T, gridsize..., nt + 2)          # save all timesteps (from -1 to nt+1 so nt+2)
            #     checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
            #     checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
            #     checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            # end
            # # Save first 2 timesteps in save buffer
            # save_buffer[fill(Colon(), N)..., 1] .= pold
            # save_buffer[fill(Colon(), N)..., 2] .= pcur
        end

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? [zeros(T, gridsize..., div(nt, snapevery)) for _ in 1:N] : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        return new{T, N}(
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

# Specific functions for ElasticIsoCPMLWaveSimulation

@views function reset!(model::ElasticIsoCPMLWaveSimulation{T, N}) where {T, N}

    # Reset computational arrays
    for p in propertynames(model.velpartic)
        getfield(model.velpartic, p) .= 0.0
    end
    for p in propertynames(model.stress)
        getfield(model.stress, p) .= 0.0
    end
    for p in propertynames(model.ψ)
        getfield(model.ψ, p) .= 0.0
    end

    # Reset gradient arrays
    if model.gradient
        for p in eachindex(model.adj)
            model.adj[p][:] .= 0.0
        end
        for p in eachindex(model.grad)
            model.grad[p] .= 0.0
        end
        for p in propertynames(model.ψ_adj)
            getfield(model.ψ_adj, p) .= 0.0
        end
    end
end
###########################################################

# Traits for ElasticIsoCPMLWaveSimulation

IsSnappableTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = CPMLBoundaryCondition()
GridTrait(::Type{<:ElasticIsoCPMLWaveSimulation}) = LocalGrid()

###########################################################

struct ElasticIsoReflWaveSimulation{T, N} <: ElasticIsoWaveSimulation{T, N} end    # TODO implementation

###########################################################

# Traits for ElasticIsoReflWaveSimulation

IsSnappableTrait(::Type{<:ElasticIsoReflWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:ElasticIsoReflWaveSimulation}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:ElasticIsoReflWaveSimulation}) = LocalGrid()

###########################################################
