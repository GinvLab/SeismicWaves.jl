###########################################################

# Functions for all AcousticWaveSimul subtypes

@views function check_courant_condition(model::AcousticWaveSimul{T,N}, vp::Array{T, N}) where {T, N}
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
    min_ppw::Int=10
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

@views function check_matprop(model::AcousticCDWaveSimul{T,N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == model.grid.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.grid.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticCDWaveSimul{T,N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    # Precompute factors
    precompute_fact!(model)
end

@views precompute_fact!(model::AcousticCDWaveSimul) = copyto!(model.grid.fields["fact"].value, (model.dt^2) .* (model.matprop.vp .^ 2))

@views function init_gradient(model::AcousticCDWaveSimul{T,N})::Dict{String, Array{T, N}} where {T, N}
    return Dict("vp" => zero(model.matprop.vp))
end

@views function accumulate_gradient!(totgrad::D, curgrad::D, ::AcousticCDWaveSimul{T,N}) where {T, N, D <: Dict{String, Array{T, N}}}
    totgrad["vp"] .+= curgrad["vp"]
end

###########################################################

struct AcousticCDCPMLWaveSimul{T, N, A <: AbstractArray{T, N}, V <: AbstractVector{T}} <: AcousticCDWaveSimul{T,N}
    # Parameters
    params::InputParametersAcoustic{T, N}
    cpmlparams::CPMLBoundaryConditionParameters{T}
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
    smooth_radius::Int
    # Snapshots
    snapevery::Union{Int, Nothing}
    snapshots::Union{Array{T}, Nothing}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::VpAcousticCDMaterialProperties{T,N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients{T, V}}
    # Checkpointing setup
    checkpointer::LinearCheckpointer{T}
    # Parallelization type
    parall::Symbol

    function AcousticCDCPMLWaveSimul(
        params::InputParametersAcoustic{T, N},
        matprop::VpAcousticCDMaterialProperties{T,N},
        cpmlparams::CPMLBoundaryConditionParameters{T};
        parall::Symbol=:threads,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5
    ) where {T, N}
        # Extract params
        nt = params.ntimesteps
        dt = params.dt
        gridspacing = params.gridspacing
        ns = params.gridsize
        halo = cpmlparams.halo
        freetop = cpmlparams.freeboundtop
        rcoef = cpmlparams.rcoef
        # Check numerics
        @assert nt > 0 "Number of timesteps must be positive!"
        @assert dt > 0 "Timestep size must be positive!"

        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        ns_cpml = freetop ? ns[1:(end-1)] : ns
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimul{T, N}, parall)
        A = backend.Data.Array{T, N}
        V = backend.Data.Array{T, 1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(ns, gridspacing)
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients{T, V}(halo, backend, true) for _ in 1:N]...)

        # Populate computational grid
        addfield!(grid, "fact" => ScalarVariableField(backend.zeros(T, ns...)))
        addfield!(grid, "pold" => ScalarVariableField(backend.zeros(T, ns...)))
        addfield!(grid, "pcur" => ScalarVariableField(backend.zeros(T, ns...)))
        addfield!(grid, "pnew" => ScalarVariableField(backend.zeros(T, ns...)))
        if gradient
            addfield!(grid, "grad_vp" => ScalarVariableField(backend.zeros(T, ns...)))
            addfield!(grid, "adjold" => ScalarVariableField(backend.zeros(T, ns...)))
            addfield!(grid, "adjcur" => ScalarVariableField(backend.zeros(T, ns...)))
            addfield!(grid, "adjnew" => ScalarVariableField(backend.zeros(T, ns...)))
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
                cat([[backend.zeros(T, [j == i ? halo + 1 : ns[j] for j in 1:N]...), backend.zeros(T, [j == i ? halo + 1 : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
            )
        )
        addfield!(
            grid,
            "ξ" => MultiVariableField(
                cat([[backend.zeros(T, [j == i ? halo : ns[j] for j in 1:N]...), backend.zeros(T, [j == i ? halo : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
            )
        )
        if gradient
            addfield!(
                grid,
                "ψ_adj" => MultiVariableField(
                    cat([[backend.zeros(T, [j == i ? halo + 1 : ns[j] for j in 1:N]...), backend.zeros(T, [j == i ? halo + 1 : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
                )
            )
            addfield!(
                grid,
                "ξ_adj" => MultiVariableField(
                    cat([[backend.zeros(T, [j == i ? halo : ns[j] for j in 1:N]...), backend.zeros(T, [j == i ? halo : ns[j] for j in 1:N]...)] for i in 1:N]...; dims=1)
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
        savecheckpoint!(checkpointer, "ψ" => grid.fields["ψ"], 0)
        savecheckpoint!(checkpointer, "ξ" => grid.fields["ξ"], 0)

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? [backend.zeros(T, ns...) for _ in 1:div(nt, snapevery)] : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        new{T, N, A, V}(
            params,
            cpmlparams,
            nt,
            dt,
            grid,
            halo,
            rcoef,
            freetop,
            gradient,
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

@views function find_nearest_grid_points(model::AcousticCDCPMLWaveSimul{T}, positions::Matrix{T})::Matrix{Int} where {T}
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

@views function check_courant_condition(model::AcousticVDStaggeredWaveSimul{T,N}, vp::Array{T, N}) where {T,N}
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
    min_ppw::Int=10
)
    # Check points per wavelength
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.gridspacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    @info "Points per wavelength: $(ppw)"
    @assert ppw >= min_ppw "Not enough points per wavelengh!"
end

@views function check_matprop(model::AcousticVDStaggeredWaveSimul{T,N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == ndims(matprop.rho) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == size(matprop.rho) == model.ns "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(size(matprop.rho)), $(model.ns)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    @assert all(matprop.rho .> 0) "Density material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

@views function update_matprop!(model::AcousticVDStaggeredWaveSimul{T,N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    copyto!(model.matprop.rho, matprop.rho)
    model.matprop.interp_method = matprop.interp_method
    # Precompute factors
    precompute_fact!(model)
end

@views function precompute_fact!(model::AcousticVDStaggeredWaveSimul{T,N}) where {T,N}
    # Precompute 1/m0 * dt factor
    copyto!(model.fact_m0, model.matprop.vp .^ 2 .* model.matprop.rho .* model.dt)
    # Precompute m1 * dt factor by interpolation
    m1_stag_interp = interpolate(1 ./ model.matprop.rho, model.matprop.interp_method)
    for i in 1:N
        copyto!(model.fact_m1_stag[i], m1_stag_interp[i] .* model.dt)
    end
end

@views function init_gradient(model::AcousticVDStaggeredWaveSimul{T,N})::Dict{String, Array{T, N}} where {T, N}
    return Dict("vp" => zero(model.matprop.vp), "rho" => zero(model.matprop.rho))
end

@views function accumulate_gradient!(totgrad::D, curgrad::D, ::AcousticVDStaggeredWaveSimul{T,N}) where {T, N, D <: Dict{String, Array{T, N}}}
    totgrad["vp"] .+= curgrad["vp"]
    totgrad["rho"] .+= curgrad["rho"]
end

###########################################################

struct AcousticVDStaggeredCPMLWaveSimul{T,N} <: AcousticVDStaggeredWaveSimul{T,N}
    # Physics
    domainextent::NTuple{N, T}
    # Numerics
    ns::NTuple{N, Int}
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
    snapshots::Union{Any, Nothing}
    # Logging parameters
    infoevery::Int
    # Gradient smoothing parameters
    smooth_radius::Int
    # Material properties
    matprop::VpRhoAcousticVDMaterialProperties{T,N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficients{T}}
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
    last_checkpoint::Union{Int, Nothing}
    save_buffer::Any
    checkpoints::Any
    checkpoints_v::Any
    checkpoints_ψ::Any
    checkpoints_ξ::Any
    # Backend
    backend::Module

    function AcousticVDStaggeredCPMLWaveSimul(
        ns::NTuple{N, Int},
        gridspacing::NTuple{N, T},
        nt::Int,
        dt::T,
        matprop::VpRhoAcousticVDMaterialProperties{T,N},
        halo::Int,
        rcoef::T;
        parall::Symbol=:threads,
        freetop::Bool=true,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5
    ) where {T,N}
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

        # Select backend
        backend = select_backend(AcousticVDStaggeredCPMLWaveSimul{T,N}, parall)
        V = backend.Data.Array{T, 1}

        # Initialize computational arrays
        fact_m0 = backend.zeros(T, ns...)
        fact_m1_stag = []
        pcur = backend.zeros(T, ns...)
        vcur = []
        for i in 1:N
            stag_ns = [ns...]
            stag_ns[i] -= 1
            push!(fact_m1_stag, backend.zeros(T, stag_ns...))
            push!(vcur, backend.zeros(T, stag_ns...))
        end
        # Initialize CPML arrays
        ψ = []
        ξ = []
        for i in 1:N
            ψ_ns = [ns...]
            ξ_ns = [ns...]
            ψ_ns[i] = halo + 1
            ξ_ns[i] = halo
            append!(ψ, [backend.zeros(T, ψ_ns...), backend.zeros(T, ψ_ns...)])
            append!(ξ, [backend.zeros(T, ξ_ns...), backend.zeros(T, ξ_ns...)])
        end
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficients{T, V}(halo, backend, true) for _ in 1:N]...)
        # Build CPML coefficient arrays for computations (they are just references to cpmlcoeffs)
        a_coeffs = []
        b_coeffs = []
        for i in 1:N
            append!(a_coeffs, [cpmlcoeffs[i].a_l, cpmlcoeffs[i].a_r, cpmlcoeffs[i].a_hl, cpmlcoeffs[i].a_hr])
            append!(b_coeffs, [cpmlcoeffs[i].b_l, cpmlcoeffs[i].b_r, cpmlcoeffs[i].b_hl, cpmlcoeffs[i].b_hr])
        end
        # Initialize gradient arrays if needed
        if gradient
            # Current gradient arrays
            curgrad_m0 = backend.zeros(T, ns...)
            curgrad_m1_stag = []
            # Adjoint arrays
            adjpcur = backend.zeros(T, ns...)
            adjvcur = []
            for i in 1:N
                stag_ns = [ns...]
                stag_ns[i] -= 1
                push!(adjvcur, backend.zeros(T, stag_ns...))
                push!(curgrad_m1_stag, backend.zeros(T, stag_ns...))
            end
            # Initialize CPML arrays
            ψ_adj = []
            ξ_adj = []
            for i in 1:N
                ψ_ns = [ns...]
                ξ_ns = [ns...]
                ψ_ns[i] = halo + 1
                ξ_ns[i] = halo
                append!(ψ_adj, [backend.zeros(T, ψ_ns...), backend.zeros(T, ψ_ns...)])
                append!(ξ_adj, [backend.zeros(T, ξ_ns...), backend.zeros(T, ξ_ns...)])
            end
            # Checkpointing setup
            if check_freq !== nothing
                @assert check_freq > 2 "Checkpointing frequency must be bigger than 2!"
                @assert check_freq < nt "Checkpointing frequency must be smaller than the number of timesteps!"
                # Time step of last checkpoint
                last_checkpoint = floor(Int, nt / check_freq) * check_freq
                # Checkpointing arrays
                save_buffer = backend.zeros(T, ns..., check_freq + 1)      # pressure window buffer
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
                        checkpoints[it] = backend.zeros(T, ns...)
                        checkpoints_v[it] = copy.(vcur)
                        checkpoints_ψ[it] = copy.(ψ)
                        checkpoints_ξ[it] = copy.(ξ)
                    end
                end
            else    # no checkpointing
                last_checkpoint = 0                                 # simulate a checkpoint at time step 0
                save_buffer = backend.zeros(T, ns..., nt + 1)          # save all timesteps (from 0 to nt so nt+1)
                checkpoints = Dict{Int, backend.Data.Array}()       # pressure checkpoints (will remain empty)
                checkpoints_v = Dict{Int, Vector{backend.Data.Array}}() # velocities checkpoints (will remain empty)
                checkpoints_ψ = Dict{Int, Any}()                    # ψ arrays checkpoints (will remain empty)
                checkpoints_ξ = Dict{Int, Any}()                    # ξ arrays checkpoints (will remain empty)
            end
            # Save timestep 0 in save buffer
            copyto!(save_buffer[fill(Colon(), N)..., 1], pcur)
        end

        # Initialize snapshots array
        snapshots = (snapevery !== nothing ? backend.zeros(T, ns..., div(nt, snapevery)) : nothing)
        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        return new{T,N}(
            domainextent,
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
