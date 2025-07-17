###########################################################

# Functions for all AcousticWaveSimulation subtypes

function check_courant_condition(model::AcousticWaveSimulation{T, N}, vp::Array{T, N}) where {T, N}
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.grid.spacing .^ 2))
    courant = vel_max * model.dt * tmp
    @info "Courant number: $(courant)"
    if model.runparams.erroronCFL
        @assert courant > 1
    elseif courant > 1
        @warn "Courant condition not satisfied! [$(courant)]"
    end
    return
end

function check_numerics(
    model::AcousticWaveSimulation{T},
    shot::ScalarShot{T};
    min_ppw::Int=10
) where {T}
    # Check points per wavelength
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.grid.spacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    
    @info "Points per wavelength: $(ppw)"
    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    if model.runparams.erroronPPW
        @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    elseif ppw >= min_ppw
        @warn "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    end
    return
end

###########################################################

# Functions for all AcousticCDWaveSimulation subtypes

function check_matprop(model::AcousticCDWaveSimulation{T, N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == model.grid.size "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.grid.size)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

function update_matprop!(model::AcousticCDWaveSimulation{T, N}, matprop::VpAcousticCDMaterialProperties{T, N}) where {T, N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    # Precompute factors
    precompute_fact!(model)
end

precompute_fact!(model::AcousticCDWaveSimulation) = copyto!(model.grid.fields["fact"].value, (model.dt^2) .* (model.matprop.vp .^ 2))

init_gradient(model::AcousticCDWaveSimulation) = Dict("vp" => zero(model.matprop.vp))

accumulate_gradient!(totgrad::D, curgrad::D, ::AcousticCDWaveSimulation{T, N}) where {T, N, D <: Dict{String, Array{T, N}}} = totgrad["vp"] .+= curgrad["vp"]

###########################################################

struct AcousticCDCPMLWaveSimulation{T, N, A <: AbstractArray{T, N}, V <: AbstractVector{T}} <: AcousticCDWaveSimulation{T, N}
    # Parameters
    params::InputParametersAcoustic{T, N}
    cpmlparams::CPMLBoundaryConditionParameters{T}
    # Numerics
    nt::Int
    dt::T
    # Computational grid
    grid::UniformFiniteDifferenceGrid{N, T}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::VpAcousticCDMaterialProperties{T, N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficientsAxis{T, V}}
    # Checkpointing setup
    checkpointer::Union{Nothing, LinearCheckpointer{T}}
    # Smooth radius for gradient
    smooth_radius::Int
    # Snapshotter setup
    snapshotter::Union{Nothing, LinearSnapshotter{T, N, Array{T, N}}}
    # Parallelization type
    parall::Symbol

    function AcousticCDCPMLWaveSimulation(
        params::InputParametersAcoustic{T, N},
        matprop::VpAcousticCDMaterialProperties{T, N},
        cpmlparams::CPMLBoundaryConditionParameters{T};
        runparams::RunParameters,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        smooth_radius::Int=0
        #parall::Symbol=:threads,
        #snapevery::Union{Int, Nothing}=nothing,
        #infoevery::Union{Int, Nothing}=nothing,
    ) where {T, N}
        # Run parameters
        parall=runparams.parall
        snapevery=runparams.snapevery
        infoevery=runparams.infoevery
        # Extract params
        nt = params.ntimesteps
        dt = params.dt
        gridspacing = params.gridspacing
        gridsize = params.gridsize
        halo = cpmlparams.halo
        freetop = cpmlparams.freeboundtop
        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        ns_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimulation{T, N}, parall)
        A = backend.Data.Array{T, N}
        V = backend.Data.Array{T, 1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(gridsize, gridspacing)
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]...)

        # Populate computational grid
        addfield!(grid, "fact" => ScalarVariableField(backend.zeros(T, gridsize...)))
        addfield!(grid, "pold" => ScalarVariableField(backend.zeros(T, gridsize...)))
        addfield!(grid, "pcur" => ScalarVariableField(backend.zeros(T, gridsize...)))
        addfield!(grid, "pnew" => ScalarVariableField(backend.zeros(T, gridsize...)))
        if gradient
            addfield!(grid, "grad_vp" => ScalarVariableField(backend.zeros(T, gridsize...)))
            addfield!(grid, "adjold" => ScalarVariableField(backend.zeros(T, gridsize...)))
            addfield!(grid, "adjcur" => ScalarVariableField(backend.zeros(T, gridsize...)))
            addfield!(grid, "adjnew" => ScalarVariableField(backend.zeros(T, gridsize...)))
        end
        # CPML memory variables
        addfield!(
            grid,
            "ψ" => MultiVariableField(  # memory variables for velocities
                [backend.zeros(T, [j == i ? 2halo : gridsize[j] for j in 1:N]...) for i in 1:N]
            )
        )
        addfield!(
            grid,
            "ξ" => MultiVariableField(  # memory variables for pressure
                [backend.zeros(T, [j == i ? 2(halo+1) : gridsize[j] for j in 1:N]...) for i in 1:N]
            )
        )
        if gradient
            # CPML memory variables
            addfield!(
                grid,
                "ψ_adj" => MultiVariableField(  # memory variables for velocities
                    [backend.zeros(T, [j == i ? 2halo : gridsize[j] for j in 1:N]...) for i in 1:N]
                )
            )
            addfield!(
                grid,
                "ξ_adj" => MultiVariableField(  # memory variables for pressure
                    [backend.zeros(T, [j == i ? 2(halo+1) : gridsize[j] for j in 1:N]...) for i in 1:N]
                )
            )
        end

        if gradient
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
        end

        if snapevery !== nothing
            # Initialize snapshotter
            snapshotter = LinearSnapshotter{Array{T, N}}(nt, snapevery, filter(p -> p.first in ["pcur"], grid.fields))
        end

        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        # Deep copy material properties
        matprop = deepcopy(matprop)

        new{T, N, A, V}(
            params,
            cpmlparams,
            nt,
            dt,
            grid,
            infoevery,
            matprop,
            cpmlcoeffs,
            gradient ? checkpointer : nothing,
            smooth_radius,
            snapevery === nothing ? nothing : snapshotter,
            parall
        )
    end
end

###########################################################

function find_nearest_grid_points(model::AcousticCDCPMLWaveSimulation{T}, positions::Matrix{T})::Matrix{Int} where {T}
    # source time functions
    nsrcs = size(positions, 1)                      # number of sources
    ncoos = size(positions, 2)                      # number of coordinates
    # find nearest grid point for each source
    idx_positions = zeros(Int, size(positions))     # sources positions (in grid points)
    for s in 1:nsrcs
        tmp = [positions[s, i] / model.grid.spacing[i] + 1 for i in 1:ncoos]
        idx_positions[s, :] .= round.(Int, tmp, RoundNearestTiesUp)
    end
    return idx_positions
end

# Specific functions for AcousticCDCPMLWaveSimulation

function reset!(model::AcousticCDCPMLWaveSimulation)
    reset!(model.grid; except=["fact"])
    if model.checkpointer !== nothing
        reset!(model.checkpointer)
    end
end

###########################################################

# Traits for AcousticCDCPMLWaveSimulation

IsSnappableTrait(::Type{<:AcousticCDCPMLWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticCDCPMLWaveSimulation}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticCDCPMLWaveSimulation}) = LocalGrid()

###########################################################

# Functions for all AcousticVDStaggeredWaveSimulation subtypes

function check_courant_condition(model::AcousticVDStaggeredWaveSimulation{T, N}, vp::Array{T, N}) where {T, N}
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt(sum(1 ./ model.grid.spacing .^ 2))
    courant = vel_max * model.dt * tmp * 7 / 6    # 7/6 comes from the higher order stencil
    @info "Courant number: $(courant)"
    if model.runparams.erroronCFL
        @assert courant > 1
    elseif courant > 1
        @warn "Courant condition not satisfied! [$(courant)]"
    end
    return
end

function check_numerics(
    model::AcousticVDStaggeredWaveSimulation{T},
    shot::ScalarShot{T};
    min_ppw::Int=10
) where {T}
    # Check points per wavelength
    vel_min = get_minimum_func(model)(model.matprop.vp)
    h_max = maximum(model.grid.spacing)
    ppw = vel_min / shot.srcs.domfreq / h_max
    
    @info "Points per wavelength: $(ppw)"
    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    if model.runparams.erroronPPW
        @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    elseif ppw >= min_ppw
        @warn "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    end
end

function check_matprop(model::AcousticVDStaggeredWaveSimulation{T, N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert ndims(matprop.vp) == ndims(matprop.rho) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(matprop.vp) == size(matprop.rho) == model.grid.size "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(size(matprop.rho)), $(model.grid.size)"
    @assert all(matprop.vp .> 0) "Pressure velocity material property must be positive!"
    @assert all(matprop.rho .> 0) "Density material property must be positive!"
    # Check courant condition
    check_courant_condition(model, matprop.vp)
end

function update_matprop!(model::AcousticVDStaggeredWaveSimulation{T, N}, matprop::VpRhoAcousticVDMaterialProperties{T, N}) where {T, N}
    # Update material properties
    copyto!(model.matprop.vp, matprop.vp)
    copyto!(model.matprop.rho, matprop.rho)
    model.matprop.interp_method = matprop.interp_method
    # Precompute factors
    precompute_fact!(model)
end

function precompute_fact!(model::AcousticVDStaggeredWaveSimulation{T, N}) where {T, N}
    # Precompute 1/m0 * dt factor
    copyto!(model.grid.fields["fact_m0"].value, model.matprop.vp .^ 2 .* model.matprop.rho .* model.dt)
    # Precompute m1 * dt factor by interpolation
    m1_stag_interp = interpolate(1 ./ model.matprop.rho, model.matprop.interp_method)
    for i in 1:N
        copyto!(model.grid.fields["fact_m1_stag"].value[i], m1_stag_interp[i] .* model.dt)
    end
end

init_gradient(model::AcousticVDStaggeredWaveSimulation) = Dict("vp" => zero(model.matprop.vp), "rho" => zero(model.matprop.rho))

function accumulate_gradient!(totgrad::D, curgrad::D, ::AcousticVDStaggeredWaveSimulation{T, N}) where {T, N, D <: Dict{String, Array{T, N}}}
    totgrad["vp"] .+= curgrad["vp"]
    totgrad["rho"] .+= curgrad["rho"]
end

###########################################################

struct AcousticVDStaggeredCPMLWaveSimulation{T, N, A <: AbstractArray{T, N}, V <: AbstractVector{T}} <: AcousticVDStaggeredWaveSimulation{T, N}
    # Parameters
    params::InputParametersAcoustic{T, N}
    cpmlparams::CPMLBoundaryConditionParameters{T}
    # Numerics
    nt::Int
    dt::T
    # Computational grid
    grid::UniformFiniteDifferenceGrid{N, T}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::VpRhoAcousticVDMaterialProperties{T, N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficientsAxis{T, V}}
    # Checkpointing setup
    checkpointer::Union{Nothing, LinearCheckpointer{T}}
    # Smooth radius for gradient
    smooth_radius::Int
    # Snapshotter setup
    snapshotter::Union{Nothing, LinearSnapshotter{T, N, Array{T, N}}}
    # Parallelization type
    parall::Symbol

    function AcousticVDStaggeredCPMLWaveSimulation(
        params::InputParametersAcoustic{T, N},
        matprop::VpRhoAcousticVDMaterialProperties{T, N},
        cpmlparams::CPMLBoundaryConditionParameters{T};
        runparams::RunParameters,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5
        #parall::Symbol=:threads,
        #snapevery::Union{Int, Nothing}=nothing,
        #infoevery::Union{Int, Nothing}=nothing,
    ) where {T, N}
          # Run parameters
        parall=runparams.parall
        snapevery=runparams.snapevery
        infoevery=runparams.infoevery
        # Extract params
        nt = params.ntimesteps
        dt = params.dt
        gridspacing = params.gridspacing
        gridsize = params.gridsize
        halo = cpmlparams.halo
        freetop = cpmlparams.freeboundtop
        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        ns_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, ns_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(AcousticCDCPMLWaveSimulation{T, N}, parall)
        A = backend.Data.Array{T, N}
        V = backend.Data.Array{T, 1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(gridsize, gridspacing)
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]...)

        # Populate computational grid
        addfield!(grid, "fact_m0" => ScalarVariableField(backend.zeros(T, gridsize...)))
        addfield!(grid, "fact_m1_stag" => MultiVariableField(
            [backend.zeros(T, (gridsize .- [i == j ? 1 : 0 for j in 1:N])...) for i in 1:N]
        ))
        addfield!(grid, "pcur" => ScalarVariableField(backend.zeros(T, gridsize...)))
        addfield!(grid, "vcur" => MultiVariableField(
            [backend.zeros(T, (gridsize .- [i == j ? 1 : 0 for j in 1:N])...) for i in 1:N]
        ))
        if gradient
            addfield!(grid, "grad_m0" => ScalarVariableField(backend.zeros(T, gridsize...)))
            addfield!(grid, "grad_m1_stag" => MultiVariableField(
                [backend.zeros(T, (gridsize .- [i == j ? 1 : 0 for j in 1:N])...) for i in 1:N]
            ))
            addfield!(grid, "adjpcur" => ScalarVariableField(backend.zeros(T, gridsize...)))
            addfield!(grid, "adjvcur" => MultiVariableField(
                [backend.zeros(T, (gridsize .- [i == j ? 1 : 0 for j in 1:N])...) for i in 1:N]
            ))
        end
        # CPML memory variables
        addfield!(
            grid,
            "ψ" => MultiVariableField(  # memory variables for velocities
                [backend.zeros(T, [j == i ? 2halo : gridsize[j] for j in 1:N]...) for i in 1:N]
            )
        )
        addfield!(
            grid,
            "ξ" => MultiVariableField(  # memory variables for pressure
                [backend.zeros(T, [j == i ? 2(halo+1) : gridsize[j] for j in 1:N]...) for i in 1:N]
            )
        )
        if gradient
            addfield!(
                grid,
                "ψ_adj" => MultiVariableField(  # memory variables for velocities
                    [backend.zeros(T, [j == i ? 2halo : gridsize[j] for j in 1:N]...) for i in 1:N]
                )
            )
            addfield!(
                grid,
                "ξ_adj" => MultiVariableField(  # memory variables for pressure
                    [backend.zeros(T, [j == i ? 2(halo+1) : gridsize[j] for j in 1:N]...) for i in 1:N]
                )
            )
        end

        if gradient
            # Initialize checkpointer
            checkpointer = LinearCheckpointer(
                nt,
                check_freq === nothing ? 1 : check_freq,
                filter(p -> p.first in ["pcur", "vcur", "ψ", "ξ"], grid.fields),
                ["pcur"];
                widths=Dict("pcur" => 1)
            )
            # Save first two timesteps
            savecheckpoint!(checkpointer, "pcur" => grid.fields["pcur"], 0)
            savecheckpoint!(checkpointer, "vcur" => grid.fields["vcur"], 0)
            savecheckpoint!(checkpointer, "ψ" => grid.fields["ψ"], 0)
            savecheckpoint!(checkpointer, "ξ" => grid.fields["ξ"], 0)
        end

        if snapevery !== nothing
            # Initialize snapshotter
            snapshotter = LinearSnapshotter{Array{N, T}}(nt, snapevery, filter(p -> p.first in ["pcur", "vcur"], grid.fields))
        end

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        # Deep copy material properties
        matprop = deepcopy(matprop)

        return new{T, N, A, V}(
            params,
            cpmlparams,
            nt,
            dt,
            grid,
            infoevery,
            matprop,
            cpmlcoeffs,
            gradient ? checkpointer : nothing,
            smooth_radius,
            snapevery === nothing ? nothing : snapshotter,
            parall
        )
    end
end

###########################################################

# Specific functions for AcousticVDStaggeredCPMLWaveSimulation

function reset!(model::AcousticVDStaggeredCPMLWaveSimulation)
    # Reset computational arrays
    reset!(model.grid; except=["fact_m0", "fact_m1_stag"])
    if model.checkpointer !== nothing
        reset!(model.checkpointer)
    end
end
###########################################################

# Traits for AcousticVDStaggeredCPMLWaveSimulation

IsSnappableTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimulation}) = Snappable()
BoundaryConditionTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimulation}) = CPMLBoundaryCondition()
GridTrait(::Type{<:AcousticVDStaggeredCPMLWaveSimulation}) = LocalGrid()

###########################################################
