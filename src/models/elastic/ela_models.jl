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

    # the following on device?
    precomp_elaprop!(model.matprop.ρ,model.matprop.μ,model.matprop.λ,
                     model.matprop.ρ_ihalf_jhalf,
                     model.matprop.μ_ihalf,model.matprop.μ_jhalf,
                     model.matprop.λ_ihalf)

    return
end


@views function scale_srctf(model::ElasticIsoWaveSimul, srctf::Matrix{<:Real}, positions::Matrix{<:Int})::Matrix{<:Real}
    scaled_tf = copy(srctf)
    return scaled_tf
end

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
    snapshots::Union{Vector{<:Array{<:Real}}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Material properties
    matprop::ElasticIsoMaterialProperty
    # # CPML coefficients
    # cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Forward computation arrays
    velpartic::Any # 2D: 2 comp, 3D: 3 comp
    stress::Any # 2D: 3 arrays, 3D: 6 arrays
    ψ::Any
    a_coeffs::Any
    b_coeffs::Any
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
        if N==2
            matprop = ElasticIsoMaterialProperty_Compute2D(λ=backend.zeros(ns...),
                                                           μ=backend.zeros(ns...),
                                                           ρ=backend.zerons(ns...),
                                                           λ_ihalf=backend.zeros((ns.-1)...),
                                                           μ_ihalf=backend.zeros((ns.-[1,0])...),
                                                           μ_jhalf=backend.zeros((ns.-[0,1])...),
                                                           ρ_ihalf_jhalf=backend.zeros((ns.-1)...),
                                                           )

        else
            error("Only elastic 2D is currently implemented.")
        end

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimul{N}, parall)

        # Initialize computational arrays
        velpartic = [backend.zeros(ns...) for _ in 1:N] # vx, vy[, vz]
        stress = [backend.zeros(ns...) for _ in 1:(N-1)*3] # σxx, σxz, σzz[, σyz, σxy, σyy]
        
        ## 2D:
        # ρ_ihalf_jhalf::Any
        # μ_ihalf::Any
        # μ_jhalf::Any
        # λ_ihalf::Any
        ##
        # Initialize CPML arrays
        ## 2D:
        # ψ_∂σxx∂x  halo
        # ψ_∂σxz∂z  halo
        # ψ_∂σxz∂x  halo
        # ψ_∂σzz∂z  halo
        # ψ_∂vx∂x   halo
        # ψ_∂vz∂z   halo
        # ψ_∂vz∂x   halo 
        # ψ_∂vx∂z   halo
        ##
        ψ = []        
        for i in 1:N
            ψ_ns = [ns...]
            ψ_ns[i] = halo 
            append!(ψ, [backend.zeros(ψ_ns...) for _ in 1:(N*2^N)])
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
            error("Gradient for elastic calculations not yet implemented!")

            # # Current gradient array
            # curgrad = backend.zeros(ns...)
            # # Adjoint arrays
            # adj = backend.zeros(ns...)
            # # Initialize CPML arrays
            # ψ_adj = []
            # ξ_adj = []
            # for i in 1:N
            #     ψ_ns = [ns...]
            #     ξ_ns = [ns...]
            #     ψ_ns[i] = halo + 1
            #     ξ_ns[i] = halo
            #     append!(ψ_adj, [backend.zeros(ψ_ns...), backend.zeros(ψ_ns...)])
            #     append!(ξ_adj, [backend.zeros(ξ_ns...), backend.zeros(ξ_ns...)])
            # end
            # # Checkpointing setup
            # if check_freq !== nothing
            #     @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
            #     @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
            #     # Time step of last checkpoint
            #     last_checkpoint = floor(Int, nt / check_freq) * check_freq
            #     # Checkpointing arrays
            #     save_buffer = backend.zeros(ns..., check_freq + 2)      # pressure window buffer
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
            #             checkpoints[it] = backend.zeros(ns...)
            #             checkpoints[it-1] = backend.zeros(ns...)
            #             checkpoints_ψ[it] = copy.(ψ)
            #             checkpoints_ξ[it] = copy.(ξ)
            #         end
            #     end
            # else    # no checkpointing
            #     last_checkpoint = 0                                 # simulate a checkpoint at time step 0 (so buffer will start from -1)
            #     save_buffer = backend.zeros(ns..., nt + 2)          # save all timesteps (from -1 to nt+1 so nt+2)
            #     checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
            #     checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
            #     checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            # end
            # # Save first 2 timesteps in save buffer
            # save_buffer[fill(Colon(), N)..., 1] .= pold
            # save_buffer[fill(Colon(), N)..., 2] .= pcur
        end

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? [zeros(ns..., div(nt, snapevery)) for _ in 1:N] : nothing)
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
            velpartic,
            stress,
            ψ,
            a_coeffs,
            b_coeffs,
            gradient ? adj : nothing,
            gradient ? ψ_adj : nothing,
            gradient ? grad : nothing,
            gradient ? last_checkpoint : nothing,
            gradient ? save_buffer : nothing,
            gradient ? checkpoints : nothing,
            gradient ? checkpoints_ψ : nothing,
            backend
        )
    end
end

###########################################################

# Specific functions for ElasticIsoCPMLWaveSimul

@views function reset!(model::ElasticIsoCPMLWaveSimul{N}) where {N}

    # Reset computational arrays
    for p in eachindex(model.velpartics)
        model.velpartic[p] .= 0.0
    end
    for p in eachindex(model.stress)
        model.stress[p] .= 0.0
    end
    for p in eachindex(model.ψ)
        model.ψ[p] .= 0.0
    end

    # Reset gradient arrays
    if model.gradient
        for p in eachindex(model.adj)
            model.adj[p][:] .= 0.0
        end
        for p in eachindex(model.grad)
            model.grad[p] .= 0.0
        end
        for p in eachindex(model.ψ_adj)
            model.ψ_adj[p] .= 0.0
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
