
###########################################################

struct AcousticCDCPMLWaveSimul{N} <: AcousticCDWaveSimul{N}
    # Physics
    ls::NTuple{N, <:Integer}
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
    matprop::VpAcousticCDMaterialProperty
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Forward computation arrays
    fact::Any
    pold::Any
    pcur::Any
    pnew::Any
    ψ::Any
    ξ::Any
    a_coeffs::Any
    b_K_coeffs::Any
    # Gradient computation arrays
    curgrad::Any
    adjold::Any
    adjcur::Any
    adjnew::Any
    ψ_adj::Any
    ξ_adj::Any
    # Checkpointing setup
    last_checkpoint::Union{<:Integer, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_ψ::Any
    checkpoints_ξ::Any
    # Backend
    backend::Module

    function AcousticCDCPMLWaveSimul{N}(
        ns::NTuple{N, <:Integer},
        gridspacing::NTuple{N, <:Real},
        nt::Integer,
        dt::Real,
        halo::Integer,
        rcoef::Real,
        parall::Symbol;
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
        matprop = VpAcousticCDMaterialProperty(zeros(ns...))

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimul{N}, parall)

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
        b_K_coeffs = []
        for i in 1:N
            append!(a_coeffs, [cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr])
            append!(b_K_coeffs, [cpmlcoeffs[i].b_K_l, cpmlcoeffs[i].b_K_r, cpmlcoeffs[i].b_K_hl, cpmlcoeffs[i].b_K_hr])
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
                last_checkpoint = 1                                 # simulate a checkpoint at time step 1 (so buffer will start from 0)
                save_buffer = backend.zeros(ns..., nt + 2)          # save all timesteps (from 0 to n+1 so n+2)
                checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
                checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
                checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
                # Save time step 0 into buffer
                save_buffer[fill(Colon(), N)..., 1] .= pcur
            end
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
            b_K_coeffs,
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

@views function check_matprop!(model::AcousticCDWaveSimul{N}, matprop::VpAcousticCDMaterialProperty{N}) where {N}
    # Checks
    @assert ndims(matprop.vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == model.ns "Material property number of grid points must be the same as the wavesim!"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop)
end

@views function update_matprop!(model::AcousticCDWaveSimul, matprop::VpAcousticCDMaterialProperty)
    # Update material properties
    model.matprop.vp .= matprop.vp
    # Precompute factors
    precompute_fact!(model)
end

@views function reset!(model::AcousticCDWaveSimul{N}) where {N}
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

# Traits for AcousticCDCPMLWaveSimul
IsSnappableTrait(::Type{<:AcousticCDCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticCDCPMLWaveSimul}) = LocalGrid()

#######################################################################

struct AcousticCDReflWaveSimul{N} <: AcousticCDWaveSimul{N} end    # TODO implementation

IsSnappableTrait(::Type{<:AcousticCDReflWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDReflWaveSimul}) = ReflectiveBoundaryCondition()
GridTrait(::Type{<:AcousticCDReflWaveSimul}) = LocalGrid()
