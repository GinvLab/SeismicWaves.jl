###########################################################

# Functions for all ElasticIsoWaveSimul subtypes

@views function check_matprop(model::ElasticIsoWaveSimul{N}, matprop::ElasticIsoMaterialProperty{N}) where {N}
    # Checks
    vp = sqrt((matprop.λ .+ 2.*matprop.μ) ./ matprop.ρ)
    @assert ndims(vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(vp) == model.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.ns)"
    @assert all(matprop.λ .> 0) "Lamè coefficient λ must be positive!"
    @assert all(matprop.μ .> 0) "Lamè coefficient μ must be positive!"
    @assert all(matprop.ρ .> 0) "Density must be positive!"
    
    # Check courant condition
    vp_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp
    @debug "Courant number: $(courant)"
    if courant > 1.0
        @warn "Courant condition not satisfied! [$(courant)]"
    end

    return
end

function check_numerics(
    model::ElasticIsoWaveSimul,
    shot::Shot;
    min_ppw::Integer=10
)
    # Check points per wavelengh
    vel_min = get_minimum_func(model)(sqrt(model.matprop.μ ./ model.matprop.ρ)) # min Vs
    h_max = maximum(model.gridspacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    @debug "Points per wavelength: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function update_matprop!(model::ElasticIsoWaveSimul{N}, matprop::ElasticIsoMaterialProperty{N}) where {N}
    # Update material properties
    model.matprop.λ .= matprop.λ
    model.matprop.μ .= matprop.μ
    model.matprop.ρ .= matprop.ρ
    return
end


# @views function scale_srctf(model::ElasticIsoWaveSimul, srctf::Matrix{<:Real}, positions::Matrix{<:Int})::Matrix{<:Real}
#     # scale with boxcar and timestep size
#     scaled_tf = srctf ./ prod(model.gridspacing) .* (model.dt^2)
#     # scale with velocity squared at each source position
#     for s in axes(scaled_tf, 2)
#         scaled_tf[:, s] .*= model.matprop.vp[positions[s, :]...] .^ 2
#     end
#     return scaled_tf
# end

###########################################################

struct ElasticIsoCPMLWaveSimul{N} <: ElasticIsoWaveSimul{N}
    # Physics
    ls::NTuple{N, <:Real}
    # Numerics
    ns::NTuple{N, <:Integer}
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
    snapshots::Union{<:Array{<:Real}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Material properties
    matprop::VpElasticIsoMaterialProperty
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Forward computation arrays
    vx::Any
    vz::Any
    σxx::Any
    σzz::Any
    σxz::Any
    λ::Any
    μ::Any
    ρ::Any
    # ρ_ihalf_jhalf::Any
    # μ_ihalf::Any
    # μ_jhalf::Any
    # λ_ihalf::Any
    ψ::Elaψ
    # ψ_∂σxx∂x::Any
    # ψ_∂σxz∂z::Any
    # ψ_∂σxz∂x::Any
    # ψ_∂σzz∂z::Any
    # ψ_∂vx∂x::Any
    # ψ_∂vz∂z::Any
    # ψ_∂vz∂x::Any
    # ψ_∂vx∂z::Any
    a_coeffs::Any
    b_coeffs::Any
    # Gradient computation arrays
    grad::Any
    adj::Any
    ψ_∂σxx∂x_adj::Any
    ψ_∂σxz∂z_adj::Any
    ψ_∂σxz∂x_adj::Any
    ψ_∂σzz∂z_adj::Any
    ψ_∂vx∂x_adj::Any
    ψ_∂vz∂z_adj::Any
    ψ_∂vz∂x_adj::Any
    ψ_∂vx∂z_adj::Any
    # Checkpointing setup
    last_checkpoint::Union{<:Integer, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_ψ_∂σxx∂x::Any
    checkpoints_ψ_∂σxz∂z::Any
    checkpoints_ψ_∂σxz∂x::Any
    checkpoints_ψ_∂σzz∂z::Any
    checkpoints_ψ_∂vx∂x::Any
    checkpoints_ψ_∂vz∂z::Any
    checkpoints_ψ_∂vz∂x::Any
    checkpoints_ψ_∂vx∂z::Any
    a_coeffs::Any
    b_coeffs::Any

    
    # Backend
    backend::Module

    function ElasticIsoCPMLWaveSimul{N}(
        ns::NTuple{N, <:Integer},
        gridspacing::NTuple{N, <:Real},
        nt::Integer,
        dt::Real,
        halo::Integer,
        rcoef::Real;
        parall::Symbol=:threads,
        freetop::Bool=true,
        gradient::Bool=false,
        check_freq::Union{<:Integer, Nothing}=nothing,
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
        # Initialize material properties
        matprop = VpElasticIsoMaterialProperty(zeros(ns...))

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimul{N}, parall)

        # Initialize computational arrays
        fact = backend.zeros(ns...)
        pold = backend.zeros(ns...)
        pcur = backend.zeros(ns...)
        pnew = backend.zeros(ns...)
        # Initialize CPML arrays
        ψ = []
        ξ = []
        for i in 1:N
            ψ_ns = [ns...]
            ξ_ns = [ns...]
            ψ_ns[i] = halo + 1
            ξ_ns[i] = halo
            append!(ψ, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
            append!(ξ, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
        end
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients(halo, backend) for _ in 1:N]...)
        # Build CPML coefficient arrays for computations (they are just references to cpmlcoeffs)
        a_coeffs = []
        b_coeffs = []
        for i in 1:N
            append!(a_coeffs, [cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr])
            append!(b_coeffs, [cpmlcoeffs[i].b_l, cpmlcoeffs[i].b_r, cpmlcoeffs[i].b_hl, cpmlcoeffs[i].b_hr])
        end
        # Initialize gradient arrays if needed
        if gradient
            # Current gradient array
            curgrad = backend.zeros(ns...)
            # Adjoint arrays
            adjold = backend.zeros(ns...)
            adjcur = backend.zeros(ns...)
            adjnew = backend.zeros(ns...)
            # Initialize CPML arrays
            ψ_adj = []
            ξ_adj = []
            for i in 1:N
                ψ_ns = [ns...]
                ξ_ns = [ns...]
                ψ_ns[i] = halo + 1
                ξ_ns[i] = halo
                append!(ψ_adj, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
                append!(ξ_adj, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
            end
            # Checkpointing setup
            if check_freq !== nothing
                @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
                @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
                # Time step of last checkpoint
                last_checkpoint = floor(Int, nt / check_freq) * check_freq
                # Checkpointing arrays
                save_buffer = backend.zeros(ns..., check_freq + 2)      # pressure window buffer
                checkpoints = Dict{Int, backend.Data.Array}()           # pressure checkpoints
                checkpoints_ψ = Dict{Int, Any}()                        # ψ arrays checkpoints
                checkpoints_ξ = Dict{Int, Any}()                        # ξ arrays checkpoints
                # Save initial conditions as first checkpoint
                checkpoints[-1] = copy(pold)
                checkpoints[0] = copy(pcur)
                checkpoints_ψ[0] = copy.(ψ)
                checkpoints_ξ[0] = copy.(ξ)
                # Preallocate future checkpoints
                for it in 1:(nt+1)
                    if it % check_freq == 0
                        checkpoints[it] = backend.zeros(ns...)
                        checkpoints[it-1] = backend.zeros(ns...)
                        checkpoints_ψ[it] = copy.(ψ)
                        checkpoints_ξ[it] = copy.(ξ)
                    end
                end
            else    # no checkpointing
                last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
                save_buffer = backend.zeros(ns..., nt + 2)          # save all timesteps (from -1 to nt+1 so nt+2)
                checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
                checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
                checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            end
            # Save first 2 timesteps in save buffer
            save_buffer[fill(Colon(), N)..., 1] .= pold
            save_buffer[fill(Colon(), N)..., 2] .= pcur
        end

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? zeros(ns..., div(nt, snapevery)) : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        return new(
            ls,
            ns,
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
            cpmlcoeffs,
            fact,
            pold,
            pcur,
            pnew,
            ψ,
            ξ,
            a_coeffs,
            b_coeffs,
            gradient ? curgrad : nothing,
            gradient ? adjold : nothing,
            gradient ? adjcur : nothing,
            gradient ? adjnew : nothing,
            gradient ? ψ_adj : nothing,
            gradient ? ξ_adj : nothing,
            gradient ? last_checkpoint : nothing,
            gradient ? save_buffer : nothing,
            gradient ? checkpoints : nothing,
            gradient ? checkpoints_ψ : nothing,
            gradient ? checkpoints_ξ : nothing,
            backend
        )
    end
end

###########################################################

# Specific functions for ElasticIsoCPMLWaveSimul

@views function reset!(model::ElasticIsoCPMLWaveSimul{N}) where {N}
    # Reset computational arrays
    model.pold .= 0.0
    model.pcur .= 0.0
    model.pnew .= 0.0
    for i in eachindex(model.ψ)
        model.ψ[i] .= 0.0
    end
    for i in eachindex(model.ξ)
        model.ξ[i] .= 0.0
    end
    # Reset gradient arrays
    if model.gradient
        model.curgrad .= 0.0
        model.adjold .= 0.0
        model.adjcur .= 0.0
        model.adjnew .= 0.0
        for i in eachindex(model.ψ_adj)
            model.ψ_adj[i] .= 0.0
        end
        for i in eachindex(model.ξ_adj)
            model.ξ_adj[i] .= 0.0
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
