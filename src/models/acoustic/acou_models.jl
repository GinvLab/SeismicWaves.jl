###########################################################

# Functions for all AcousticWaveSimul subtypes

@views function check_courant_condition(model::AcousticWaveSimul{N}, vp::Array{<:Real, N}) where {N}
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp
    @debug "Courant number: $(courant)"
    if courant > 1.0
        @warn "Courant condition not satisfied! [$(courant)]"
    end
end

function check_numerics(
    model::AcousticWaveSimul,
    shot::Shot;
    min_ppw::Integer=10
)
    # Check points per wavelength
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.gridspacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    @debug "Points per wavelength: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

###########################################################

# Functions for all AcousticCDWaveSimul subtypes

@views function scale_srctf(model::AcousticCDWaveSimul, srctf::Matrix{<:Real}, positions::Matrix{<:Int})::Matrix{<:Real}
    # scale with boxcar and timestep size
    scaled_tf = srctf ./ prod(model.gridspacing) .* (model.dt^2)
    # scale with velocity squared at each source position
    for s in axes(scaled_tf, 2)
        scaled_tf[:, s] .*= model.matprop.vp[positions[s, :]...] .^ 2
    end
    return scaled_tf
end

@views function check_matprop(model::AcousticCDWaveSimul{N}, matprop::VpAcousticCDMaterialProperty{N}) where {N}
    # Checks
    @assert ndims(matprop.vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == model.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticCDWaveSimul{N}, matprop::VpAcousticCDMaterialProperty{N}) where {N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    # Precompute factors
    precompute_fact!(model)
end

@views precompute_fact!(model::AcousticCDWaveSimul) = copyto!(model.fact, (model.dt^2) .* (model.matprop.vp .^ 2))

###########################################################

struct AcousticCDCPMLWaveSimul{N} <: AcousticCDWaveSimul{N}
    # Physics
    domainextent::NTuple{N, <:Real}
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
    totgrad_size::Union{Vector{<:Integer}, Nothing}
    check_freq::Union{<:Integer, Nothing}
    # Snapshots
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{<:Array{<:Real}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Gradient smoothing parameters
    smooth_radius::Integer
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
    b_coeffs::Any
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
    parall::Symbol
    
    function AcousticCDCPMLWaveSimul{N}(
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
        infoevery::Union{<:Integer, Nothing}=nothing,
        smooth_radius::Integer=5
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
        domainextent = gridspacing .* (ns .- 1)
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
        cpmlcoeffs = tuple([CPMLCoefficients(halo, backend, sizehalfgrdplusone=true) for _ in 1:N]...)
        # Build CPML coefficient arrays for computations (they are just references to cpmlcoeffs)
        a_coeffs = []
        b_coeffs = []
        for i in 1:N
            append!(a_coeffs, [cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr])
            append!(b_coeffs, [cpmlcoeffs[i].b_l, cpmlcoeffs[i].b_r, cpmlcoeffs[i].b_hl, cpmlcoeffs[i].b_hr])
        end
        # Initialize gradient arrays if needed
        if gradient
            totgrad_size = [ns...]
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
            copyto!(save_buffer[fill(Colon(), N)..., 1], pold)
            copyto!(save_buffer[fill(Colon(), N)..., 2], pcur)
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
            domainextent,
            ns,
            gridspacing,
            nt,
            dt,
            halo,
            rcoef,
            freetop,
            gradient,
            gradient ? totgrad_size : nothing,
            gradient ? check_freq : nothing,
            snapevery,
            snapshots,
            infoevery,
            smooth_radius,
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
            backend,
            parall
        )
    end
end

###########################################################

# Specific functions for AcousticCDCPMLWaveSimul

@views function reset!(model::AcousticCDCPMLWaveSimul{N}) where {N}
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

# Traits for AcousticCDCPMLWaveSimul

IsSnappableTrait(::Type{<:AcousticCDCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticCDCPMLWaveSimul}) = LocalGrid()

###########################################################

# Functions for all AcousticVDStaggeredWaveSimul subtypes

@views function check_courant_condition(model::AcousticVDStaggeredWaveSimul{N}, vp::Array{<:Real, N}) where {N}
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp * 7/6    # 7/6 comes from the higher order stencil
    @debug "Courant number: $(courant)"
    if courant > 1
        @warn "Courant condition not satisfied! [$(courant)]"
    end
end

@views function scale_srctf(model::AcousticVDStaggeredWaveSimul, srctf::Matrix{<:Real}, positions::Matrix{<:Int})::Matrix{<:Real}
    # scale with boxcar and timestep size
    scaled_tf = srctf ./ prod(model.gridspacing) .* (model.dt)
    # scale with velocity squared times density at each source position (its like dividing by m0)
    for s in axes(scaled_tf, 2)
        scaled_tf[:, s] .*= model.matprop.vp[positions[s, :]...] ^ 2 * model.matprop.rho[positions[s, :]...]
    end
    return scaled_tf
end

@views function check_matprop(model::AcousticVDStaggeredWaveSimul{N}, matprop::VpRhoAcousticVDMaterialProperty{N}) where {N}
    # Checks
    @assert ndims(matprop.vp) == ndims(matprop.rho) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == size(matprop.rho) == model.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(size(matprop.rho)), $(model.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    @assert all(matprop.rho .> 0) "Density material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticVDStaggeredWaveSimul{N}, matprop::VpRhoAcousticVDMaterialProperty{N}) where {N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    copyto!(model.matprop.rho, matprop.rho)
    model.matprop.interp_method = matprop.interp_method
    # Precompute factors
    precompute_fact!(model)
end

@views function precompute_fact!(model::AcousticVDStaggeredWaveSimul{N}) where {N}
    # Precompute 1/m0 * dt factor
    copyto!(model.fact_m0, model.matprop.vp .^ 2 .* model.matprop.rho .* model.dt)
    # Precompute m1 * dt factor by interpolation
    m1_stag_interp = interpolate(1 ./ model.matprop.rho, model.matprop.interp_method)
    for i in 1:N
        copyto!(model.fact_m1_stag[i], m1_stag_interp[i] .* model.dt)
    end
end

###########################################################

struct AcousticVDStaggeredCPMLWaveSimul{N} <: AcousticVDStaggeredWaveSimul{N}
    # Physics
    domainextent::NTuple{N, <:Real}
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
    totgrad_size::Union{Vector{<:Integer}, Nothing}
    check_freq::Union{<:Integer, Nothing}
    # Snapshots
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{<:Array{<:Real}, Nothing}
    # Logging parameters
    infoevery::Integer
    # Gradient smoothing parameters
    smooth_radius::Integer
    # Material properties
    matprop::VpRhoAcousticVDMaterialProperty{N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Forward computation arrays
    fact_m0::Any
    fact_m1_stag::Any
    pcur::Any
    vcur::Any
    ψ::Any
    ξ::Any
    a_coeffs::Any
    b_coeffs::Any
    # Gradient computation arrays
    curgrad_m0::Any
    curgrad_m1_stag::Any
    adjpcur::Any
    adjvcur::Any
    ψ_adj::Any
    ξ_adj::Any
    # Checkpointing setup
    last_checkpoint::Union{<:Integer, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_v::Any
    checkpoints_ψ::Any
    checkpoints_ξ::Any
    # Backend
    backend::Module

    function AcousticVDStaggeredCPMLWaveSimul{N}(
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
        infoevery::Union{<:Integer, Nothing}=nothing,
        smooth_radius::Integer=5
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
        domainextent = gridspacing .* (ns .- 1)
        # Initialize material properties
        matprop = VpRhoAcousticVDMaterialProperty(zeros(ns...), zeros(ns...))

        # Select backend
        backend = select_backend(AcousticVDStaggeredCPMLWaveSimul{N}, parall)

        # Initialize computational arrays
        fact_m0 = backend.zeros(ns...)
        fact_m1_stag = []
        pcur = backend.zeros(ns...)
        vcur = []
        for i in 1:N
            stag_ns = [ns...]
            stag_ns[i] -= 1
            push!(fact_m1_stag, backend.zeros(stag_ns...))
            push!(vcur, backend.zeros(stag_ns...))
        end
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
            totgrad_size = [ns..., 2]
            # Current gradient arrays
            curgrad_m0 = backend.zeros(ns...)
            curgrad_m1_stag = []
            # Adjoint arrays
            adjpcur = backend.zeros(ns...)
            adjvcur = []
            for i in 1:N
                stag_ns = [ns...]
                stag_ns[i] -= 1
                push!(adjvcur, backend.zeros(stag_ns...))
                push!(curgrad_m1_stag, backend.zeros(stag_ns...))
            end
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
                save_buffer = backend.zeros(ns..., check_freq + 1)      # pressure window buffer
                checkpoints = Dict{Int, backend.Data.Array}()           # pressure checkpoints
                checkpoints_v = Dict{Int, Vector{backend.Data.Array}}() # velocities checkpoints
                checkpoints_ψ = Dict{Int, Any}()                        # ψ arrays checkpoints
                checkpoints_ξ = Dict{Int, Any}()                        # ξ arrays checkpoints
                # Save initial conditions as first checkpoint
                checkpoints[0] = copy(pcur)
                checkpoints_v[0] = copy.(vcur)
                checkpoints_ψ[0] = copy.(ψ)
                checkpoints_ξ[0] = copy.(ξ)
                # Preallocate future checkpoints
                for it in 1:(nt+1)
                    if it % check_freq == 0
                        checkpoints[it] = backend.zeros(ns...)
                        checkpoints_v[it] = copy.(vcur)
                        checkpoints_ψ[it] = copy.(ψ)
                        checkpoints_ξ[it] = copy.(ξ)
                    end
                end
            else    # no checkpointing
                last_checkpoint = 0                                 # simulate a checkpoint at time step 0
                save_buffer = backend.zeros(ns..., nt + 1)          # save all timesteps (from 0 to nt so nt+1)
                checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
                checkpoints_v = Dict{Int, Vector{backend.Data.Array}}() # velocities checkpoints (will remain empty)
                checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
                checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            end
            # Save timestep 0 in save buffer
            copyto!(save_buffer[fill(Colon(), N)..., 1], pcur)
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
            domainextent,
            ns,
            gridspacing,
            nt,
            dt,
            halo,
            rcoef,
            freetop,
            gradient,
            gradient ? totgrad_size : nothing,
            gradient ? check_freq : nothing,
            snapevery,
            snapshots,
            infoevery,
            smooth_radius,
            matprop,
            cpmlcoeffs,
            fact_m0,
            fact_m1_stag,
            pcur,
            vcur,
            ψ,
            ξ,
            a_coeffs,
            b_coeffs,
            gradient ? curgrad_m0 : nothing,
            gradient ? curgrad_m1_stag : nothing,
            gradient ? adjpcur : nothing,
            gradient ? adjvcur : nothing,
            gradient ? ψ_adj : nothing,
            gradient ? ξ_adj : nothing,
            gradient ? last_checkpoint : nothing,
            gradient ? save_buffer : nothing,
            gradient ? checkpoints : nothing,
            gradient ? checkpoints_v : nothing,
            gradient ? checkpoints_ψ : nothing,
            gradient ? checkpoints_ξ : nothing,
            backend
        )
    end
end

###########################################################

# Specific functions for AcousticVDStaggeredCPMLWaveSimul

@views function reset!(model::AcousticVDStaggeredCPMLWaveSimul)
    # Reset computational arrays
    model.pcur .= 0.0
    for i in eachindex(model.vcur)
        model.vcur[i] .= 0.0
    end
    for i in eachindex(model.ψ)
        model.ψ[i] .= 0.0
    end
    for i in eachindex(model.ξ)
        model.ξ[i] .= 0.0
    end
    # Reset gradient arrays
    if model.gradient
        model.curgrad_m0 .= 0.0
        for i in eachindex(model.curgrad_m1_stag)
            model.curgrad_m1_stag[i] .= 0.0
        end
        model.adjpcur .= 0.0
        for i in eachindex(model.adjvcur)
            model.adjvcur[i] .= 0.0
        end
        for i in eachindex(model.ψ_adj)
            model.ψ_adj[i] .= 0.0
        end
        for i in eachindex(model.ξ_adj)
            model.ξ_adj[i] .= 0.0
        end
    end
end
###########################################################

# Traits for AcousticVDStaggeredCPMLWaveSimul

IsSnappableTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimul}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimul}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimul}) = LocalGrid()

###########################################################
