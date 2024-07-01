###########################################################

# Functions for all AcousticWaveSimul subtypes

@views function check_courant_condition(model::AcousticWaveSimul{N}, vp::Array{<:Real, N}) where {N}
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.grid.gridspacing .^ 2))
    courant = vel_max * model.dt * tmp
    @info "Courant number: $(courant)"
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
    h_max = maximum(model.grid.gridspacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    @info "Points per wavelength: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

###########################################################

# Functions for all AcousticCDWaveSimul subtypes

@views function check_matprop(model::AcousticCDWaveSimul{N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == model.grid.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.grid.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticCDWaveSimul{N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    # Precompute factors
    precompute_fact!(model)
end

@views precompute_fact!(model::AcousticCDWaveSimul) = copyto!(model.grid.fields["fact"].value, (model.dt^2) .* (model.matprop.vp .^ 2))

###########################################################

struct AcousticCDCPMLWaveSimul{T, N, A <: AbstractArray{T, N}} <: AcousticCDWaveSimul{N}
    # Numerics
    nt::Int
    dt::T
    # Computational grid
    grid::UniformFiniteDifferenceGrid{N, T}
    # BDC and CPML parameters
    halo::Int
    rcoef::T
    freetop::Bool
    # Gradient computation setup
    gradient::Bool
    totgrad_size::Union{Vector{Int}, Nothing}
    smooth_radius::Int
    # Snapshots
    snapevery::Union{Int, Nothing}
    snapshots::Union{Vector{A}, Nothing}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::VpAcousticCDMaterialProperties
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients}
    # Checkpointing setup
    checkpointer::LinearCheckpointer{T}
    # Parallelization type
    parall::Symbol

    function AcousticCDCPMLWaveSimul(
        ns::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        nt::Int,
        dt::T,
        halo::Int,
        rcoef::T;
        parall::Symbol=:threads,
        freetop::Bool=true,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5
    ) where {T, N}
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"

        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        ns_cpml = freetop ? ns[1:(end-1)] : ns
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimul{T, N}, parall)
        A = backend.Data.Array{N}
        V = backend.Data.Array{1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(ns, gridspacing)
        # Initialize material properties
        matprop = VpAcousticCDMaterialProperties(zeros(ns...))
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients(halo, backend, true) for _ in 1:N]...)

        # Populate computational grid
        addfield!(grid, "fact" => ScalarVariableField(backend.zeros(ns...)))
        addfield!(grid, "pold" => ScalarVariableField(backend.zeros(ns...)))
        addfield!(grid, "pcur" => ScalarVariableField(backend.zeros(ns...)))
        addfield!(grid, "pnew" => ScalarVariableField(backend.zeros(ns...)))
        if gradient
            addfield!(grid, "grad_vp" => ScalarVariableField(backend.zeros(ns...)))
            addfield!(grid, "adjold" => ScalarVariableField(backend.zeros(ns...)))
            addfield!(grid, "adjcur" => ScalarVariableField(backend.zeros(ns...)))
            addfield!(grid, "adjnew" => ScalarVariableField(backend.zeros(ns...)))
        end
        # CPML coefficients
        addfield!(
            grid,
            "a_pml" => MultiVariableField(
                cat([[cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr] for i in 1:N]...; dims=1)
            )
        )
        addfield!(
            grid,
            "b_pml" => MultiVariableField(
                cat([[cpmlcoeffs[i].b_l, cpmlcoeffs[i].b_r, cpmlcoeffs[i].b_hl, cpmlcoeffs[i].b_hr] for i in 1:N]...; dims=1)
            )
        )
        # CPML memory variables
        addfield!(
            grid,
            "ψ" => MultiVariableField(
                cat([[backend.zeros([j == i ? halo + 1 : ns[j] for j in 1:N]...), backend.zeros([j == i ? halo + 1 : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
            )
        )
        addfield!(
            grid,
            "ξ" => MultiVariableField(
                cat([[backend.zeros([j == i ? halo : ns[j] for j in 1:N]...), backend.zeros([j == i ? halo : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
            )
        )
        if gradient
            addfield!(
                grid,
                "ψ_adj" => MultiVariableField(
                    cat([[backend.zeros([j == i ? halo + 1 : ns[j] for j in 1:N]...), backend.zeros([j == i ? halo + 1 : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
                )
            )
            addfield!(
                grid,
                "ξ_adj" => MultiVariableField(
                    cat([[backend.zeros([j == i ? halo : ns[j] for j in 1:N]...), backend.zeros([j == i ? halo : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
                )
            )
        end

        # Initialize checkpointer
        checkpointer = LinearCheckpointer(
            nt,
            check_freq === nothing ? 1 : check_freq,
            filter(p -> p.first in ["pcur", "ψ", "ξ"], grid.fields),
            ["pcur"];
            widths=Dict("pcur" => 2)
        )
        # Save first two timesteps
        savecheckpoint!(checkpointer, "pcur" => grid.fields["pold"], -1)
        savecheckpoint!(checkpointer, "pcur" => grid.fields["pcur"], 0)

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? [backend.zeros(ns...) for _ in 1:div(nt, snapevery)] : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new{T, N, A}(
            nt,
            dt,
            grid,
            halo,
            rcoef,
            freetop,
            gradient,
            gradient ? [ns...] : nothing,
            smooth_radius,
            snapevery,
            snapshots,
            infoevery,
            matprop,
            cpmlcoeffs,
            checkpointer,
            parall
        )
    end
end

###########################################################

@views function find_nearest_grid_points(model::AcousticCDCPMLWaveSimul, positions::Matrix{<:Real})::Matrix{<:Int}
    # source time functions
    nsrcs = size(positions, 1)                      # number of sources
    ncoos = size(positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    idx_positions = zeros(Int, size(positions))     # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [positions[s, i] / model.grid.gridspacing[i] + 1 for i in 1:ncoos]
        idx_positions[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    return idx_positions
end

# Specific functions for AcousticCDCPMLWaveSimul

@views function reset!(model::AcousticCDCPMLWaveSimul)
    reset!(model.grid; except=["fact", "a_pml", "b_pml"])
    reset!(model.checkpointer)
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
    courant = vel_max * model.dt * tmp * 7 / 6    # 7/6 comes from the higher order stencil
    @info "Courant number: $(courant)"
    if courant > 1
        @warn "Courant condition not satisfied! [$(courant)]"
    end
end

function check_numerics(
    model::AcousticVDStaggeredWaveSimul,
    shot::Shot;
    min_ppw::Integer=10
)
    # Check points per wavelength
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.gridspacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    @info "Points per wavelength: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_matprop(model::AcousticVDStaggeredWaveSimul{N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == ndims(matprop.rho) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == size(matprop.rho) == model.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(size(matprop.rho)), $(model.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    @assert all(matprop.rho .> 0) "Density material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticVDStaggeredWaveSimul{N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
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
    totgrad_size::Union{Vector{Int}, Nothing}
    check_freq::Union{<:Integer, Nothing}
    # Snapshots
    snapevery::Union{<:Integer, Nothing}
    snapshots::Union{Any, Nothing}
    # Logging parameters
    infoevery::Integer
    # Gradient smoothing parameters
    smooth_radius::Integer
    # Material properties
    matprop::VpRhoAcousticVDMaterialProperties
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
        matprop = VpRhoAcousticVDMaterialProperties(zeros(ns...), zeros(ns...))

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
        cpmlcoeffs = tuple([CPMLCoefficients(halo, backend, true) for _ in 1:N]...)
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
        snapshots = (snapevery !== nothing ? backend.zeros(ns..., div(nt, snapevery)) : nothing)
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
