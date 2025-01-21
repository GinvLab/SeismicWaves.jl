###########################################################

# Functions for all ElasticIsoWaveSimulation subtypes

# Scaling for ElasticIsoWaveSimulation
function possrcrec_scaletf(model::ElasticIsoWaveSimulation{T}, shot::ExternalForceShot{T, 2}; sincinterp=false) where {T}
    if sincinterp
        # interpolation coefficients for sources in vx
        nsrcs = size(shot.srcs.positions, 1)
        srccoeij_vx, srccoeval_vx = spread_positions(model.grid, shot.srcs.positions; shift=(model.grid.spacing[1]/2, 0.0), mirror=false)
        # interpolation coefficients for sources in vz
        srccoeij_vz, srccoeval_vz = spread_positions(model.grid, shot.srcs.positions; shift=(0.0, model.grid.spacing[2]/2), mirror=false)
        nrecs = size(shot.recs.positions, 1)
        # interpolation coefficients for receivers in vx
        reccoeij_vx, reccoeval_vx = spread_positions(model.grid, shot.recs.positions; shift=(model.grid.spacing[1]/2, 0.0), mirror=false)
        # interpolation coefficients for receivers in vz
        reccoeij_vz, reccoeval_vz = spread_positions(model.grid, shot.recs.positions; shift=(0.0, model.grid.spacing[2]/2), mirror=false)
    else
        src_idx_positions = find_nearest_grid_points(model, shot.srcs.positions)
        rec_idx_positions = find_nearest_grid_points(model, shot.recs.positions)
        nsrcs = size(shot.srcs.positions, 1)
        nrecs = size(shot.recs.positions, 1)
        srccoeij_vx = [[src_idx_positions[s, 1] src_idx_positions[s, 2]] for s in 1:nsrcs]
        srccoeval_vx = [ones(T, 1) for _ in 1:nsrcs]
        srccoeij_vz = [[src_idx_positions[s, 1] src_idx_positions[s, 2]] for s in 1:nsrcs]
        srccoeval_vz = [ones(T, 1) for _ in 1:nsrcs]
        reccoeij_vx = [[rec_idx_positions[r, 1] rec_idx_positions[r, 2]] for r in 1:nrecs]
        reccoeval_vx = [ones(T, 1) for _ in 1:nrecs]
        reccoeij_vz = [[rec_idx_positions[r, 1] rec_idx_positions[r, 2]] for r in 1:nrecs]
        reccoeval_vz = [ones(T, 1) for _ in 1:nrecs]
    end

    return srccoeij_vx, srccoeval_vx, srccoeij_vz, srccoeval_vz, reccoeij_vx, reccoeval_vx, reccoeij_vz, reccoeval_vz, shot.srcs.tf
end

function possrcrec_scaletf(model::ElasticIsoWaveSimulation{T}, shot::MomentTensorShot{T, 2}; sincinterp=false) where {T}
    if sincinterp
        nsrcs = size(shot.srcs.positions, 1)
        # interpolation coefficients for sources in σxx and σzz
        srccoeij_xx, srccoeval_xx = spread_positions(model.grid, shot.srcs.positions; shift=(0.0, 0.0), mirror=true)
        # interpolation coefficients for sources in σxz
        srccoeij_xz, srccoeval_xz = spread_positions(model.grid, shot.srcs.positions; shift=(model.grid.spacing[1]/2, model.grid.spacing[2]/2), mirror=true)
        nrecs = size(shot.recs.positions, 1)
        # interpolation coefficients for receivers in vx
        reccoeij_vx, reccoeval_vx = spread_positions(model.grid, shot.recs.positions; shift=(model.grid.spacing[1]/2, 0.0), mirror=false)
        # interpolation coefficients for receivers in vz
        reccoeij_vz, reccoeval_vz = spread_positions(model.grid, shot.recs.positions; shift=(0.0, model.grid.spacing[2]/2), mirror=false)
    else
        src_idx_positions = find_nearest_grid_points(model, shot.srcs.positions)
        rec_idx_positions = find_nearest_grid_points(model, shot.recs.positions)
        nsrcs = size(shot.srcs.positions, 1)
        nrecs = size(shot.recs.positions, 1)
        srccoeij_xx = [[src_idx_positions[s, 1] src_idx_positions[s, 2]] for s in 1:nsrcs]
        srccoeval_xx = [ones(T, 1) for _ in 1:nsrcs]
        srccoeij_xz = [[src_idx_positions[s, 1] src_idx_positions[s, 2]] for s in 1:nsrcs]
        srccoeval_xz = [ones(T, 1) for _ in 1:nsrcs]
        reccoeij_vx = [[rec_idx_positions[r, 1] rec_idx_positions[r, 2]] for r in 1:nrecs]
        reccoeval_vx = [ones(T, 1) for _ in 1:nrecs]
        reccoeij_vz = [[rec_idx_positions[r, 1] rec_idx_positions[r, 2]] for r in 1:nrecs]
        reccoeval_vz = [ones(T, 1) for _ in 1:nrecs]
    end

    return srccoeij_xx, srccoeval_xx, srccoeij_xz, srccoeval_xz, reccoeij_vx, reccoeval_vx, reccoeij_vz, reccoeval_vz, shot.srcs.tf
end



function check_matprop(model::ElasticIsoWaveSimulation{T, N}, matprop::ElasticIsoMaterialProperties{T, N}) where {T, N}
    # Checks
    @assert all(matprop.λ .>= 0) "Lamè coefficient λ must be positive!"
    @assert all(matprop.μ .>= 0) "Lamè coefficient μ must be positive!"
    @assert all(matprop.ρ .> 0) "Density must be positive!"
    vp = sqrt.((matprop.λ .+ 2.0 .* matprop.μ) ./ matprop.ρ)
    @assert ndims(vp) == N "Material property dimensionality must be the same as the wavesim!"
    @assert size(vp) == model.grid.size "Material property number of grid points must be the same as the wavesim! \n $(size(matprop.vp)), $(model.grid.size)"

    # Check courant condition
    vel_max = get_maximum_func(model)(vp)
    tmp = sqrt.(sum(1 ./ model.grid.spacing .^ 2))
    courant = vel_max * model.dt * tmp * 7 / 6  # 7/6 comes from the higher order stencil
    @info "Courant number: $(courant)"
    if courant > 1
        @warn "Courant condition not satisfied! [$(courant)]"
    end

    return
end

function check_numerics(
    model::ElasticIsoWaveSimulation{T},
    shot::Union{MomentTensorShot{T}, ExternalForceShot{T}};
    min_ppw::Int=10
) where {T}
    # Check points per wavelengh
    # min Vs
    vel_min = get_minimum_func(model)(sqrt.(model.matprop.μ ./ model.matprop.ρ))
    if vel_min == 0
        # compute minimum vp
        vel_min = get_minimum_func(model)(sqrt.((model.matprop.λ .+ 2.0 .* model.matprop.μ) ./ model.matprop.ρ))
    end
    h_max = maximum(model.grid.spacing)
    fmax = shot.srcs.domfreq * 2.0
    ppw = vel_min / (fmax * h_max)
    @info "Points per wavelength: $(ppw)"

    dh0 = round((vel_min / (min_ppw * fmax)); digits=2)
    @assert ppw >= min_ppw "Not enough points per wavelength (assuming fmax = 2*domfreq)! \n [$(round(ppw,digits=1)) instead of >= $min_ppw]\n  Grid spacing should be <= $dh0"
    return
end

function update_matprop!(model::ElasticIsoWaveSimulation{T, 2}, matprop::ElasticIsoMaterialProperties{T, 2}) where {T}

    # Update material properties
    copyto!(model.matprop.λ, matprop.λ)
    copyto!(model.matprop.μ, matprop.μ)
    copyto!(model.matprop.ρ, matprop.ρ)
    model.matprop.interp_method_ρ = matprop.interp_method_ρ
    model.matprop.interp_method_λ = matprop.interp_method_λ
    model.matprop.interp_method_μ = matprop.interp_method_μ
    # Precompute factors
    precomp_elaprop!(model)
end

function precomp_elaprop!(model::ElasticIsoWaveSimulation{T, 2}) where {T}
    # Excract fields
    λ = model.grid.fields["λ"].value
    μ = model.grid.fields["μ"].value
    ρ_ihalf = model.grid.fields["ρ_ihalf"].value
    ρ_jhalf = model.grid.fields["ρ_jhalf"].value
    μ_ihalf_jhalf = model.grid.fields["μ_ihalf_jhalf"].value
    # Copy from internal matprop
    copyto!(λ, model.matprop.λ)
    copyto!(μ, model.matprop.μ)
    #-------------------------------------------------------------
    # pre-interpolate properties at half distances between nodes
    #-------------------------------------------------------------
    copyto!(ρ_ihalf, interp(model.matprop.interp_method_ρ, model.matprop.ρ, 1))
    copyto!(ρ_jhalf, interp(model.matprop.interp_method_ρ, model.matprop.ρ, 2))
    copyto!(μ_ihalf_jhalf, interp(model.matprop.interp_method_μ, model.matprop.μ, [1, 2]))
    if model.checkpointer !== nothing
        # Interpolated material properties for adjoint evolution
        λ_ihalf = model.grid.fields["λ_ihalf"].value
        λ_jhalf = model.grid.fields["λ_jhalf"].value
        μ_ihalf = model.grid.fields["μ_ihalf"].value
        μ_jhalf = model.grid.fields["μ_jhalf"].value
        copyto!(λ_ihalf, interp(model.matprop.interp_method_λ, model.matprop.λ, 1))
        copyto!(λ_jhalf, interp(model.matprop.interp_method_λ, model.matprop.λ, 2))
        copyto!(μ_ihalf, interp(model.matprop.interp_method_μ, model.matprop.μ, 1))
        copyto!(μ_jhalf, interp(model.matprop.interp_method_μ, model.matprop.μ, 2))
    end
    return
end

##############################################################

struct ElasticIsoCPMLWaveSimulation{T, N, A <: AbstractArray{T, N}, V <: AbstractVector{T}} <: ElasticIsoWaveSimulation{T, N}
    # Parameters
    params::InputParametersElastic{T, N}
    cpmlparams::CPMLBoundaryConditionParameters{T}
    # Numerics
    nt::Int
    dt::T
    # Computational grid
    grid::UniformFiniteDifferenceGrid{N, T}
    # Logging parameters
    infoevery::Int
    # Material properties
    matprop::ElasticIsoMaterialProperties{T, N}
    # CPML coefficients
    cpmlcoeffs::NTuple{N, CPMLCoefficientsAxis{T, V}}
    # Checkpointing setup
    checkpointer::Union{Nothing, LinearCheckpointer{T}}
    # Smooth radius for gradient
    smooth_radius::Int
    # Sinc source receiver interpolation
    sincinterp::Bool
    # Snapshotter setup
    snapshotter::Union{Nothing, LinearSnapshotter{T, N, Array{T, N}}}
    # Parallelization type
    parall::Symbol

    function ElasticIsoCPMLWaveSimulation(
        params::InputParametersElastic{T, N},
        matprop::ElasticIsoMaterialProperties{T, N},
        cpmlparams::CPMLBoundaryConditionParameters{T};
        parall::Symbol=:threads,
        gradient::Bool=false,
        check_freq::Union{Int, Nothing}=nothing,
        snapevery::Union{Int, Nothing}=nothing,
        infoevery::Union{Int, Nothing}=nothing,
        smooth_radius::Int=5,
        sincinterp::Bool=true
    ) where {T, N}
        # Extract params
        nt = params.ntimesteps
        dt = params.dt
        gridspacing = params.gridspacing
        gridsize = params.gridsize
        halo = cpmlparams.halo
        freetop = cpmlparams.freeboundtop
        # Check BDC parameters
        @assert halo >= 0 "CPML halo size must be non-negative!"
        gridsize_cpml = freetop ? gridsize[1:(end-1)] : gridsize
        @assert all(n -> n >= 2halo + 3, gridsize_cpml) "Number grid points in the dimensions with C-PML boundaries must be at least 2*halo+3 = $(2halo+3)!"

        # Select backend
        backend = select_backend(ElasticIsoCPMLWaveSimulation{T, N}, parall)
        A = backend.Data.Array{T, N}
        V = backend.Data.Array{T, 1}
        # Initialize computational grid
        grid = UniformFiniteDifferenceGrid(gridsize, gridspacing)
        # Initialize CPML coefficients
        cpmlcoeffs = tuple([CPMLCoefficientsAxis{T, V}(halo, backend) for _ in 1:N]...)

        # Populate computational grid
        if N == 2
            # Stress and velocity
            addfield!(grid, "σ" => MultiVariableField(
                [
                    backend.zeros(T, gridsize...),              # σxx
                    backend.zeros(T, gridsize...),              # σzz
                    backend.zeros(T, (gridsize .- 1)...)        # σxz
                ]
            ))
            addfield!(grid, "v" => MultiVariableField(
                [
                    backend.zeros(T, (gridsize .- [1, 0])...),  # vx
                    backend.zeros(T, (gridsize .- [0, 1])...)   # vz
                ]
            ))
            # Material properties
            addfield!(grid, "λ" => ScalarVariableField(
                backend.zeros(T, gridsize...)
            ))
            addfield!(grid, "μ" => ScalarVariableField(
                backend.zeros(T, gridsize...)
            ))
            addfield!(grid, "ρ_ihalf" => ScalarVariableField(
                backend.zeros(T, (gridsize .- [1, 0])...)
            ))
            addfield!(grid, "ρ_jhalf" => ScalarVariableField(
                backend.zeros(T, (gridsize .- [0, 1])...)
            ))
            addfield!(grid, "μ_ihalf_jhalf" => ScalarVariableField(
                backend.zeros(T, (gridsize .- 1)...)
            ))
            # CPML memory variables
            addfield!(grid, "ψ_∂σ∂x" => MultiVariableField(
                [
                    backend.zeros(T, 2halo    , gridsize[2]  ), # ψ_∂σxx∂x
                    backend.zeros(T, 2(halo+1), gridsize[2]-1)  # ψ_∂σxz∂x
                ]
            ))
            addfield!(grid, "ψ_∂σ∂z" => MultiVariableField(
                [
                    backend.zeros(T, gridsize[1]  , 2halo    ), # ψ_∂σzz∂z
                    backend.zeros(T, gridsize[1]-1, 2(halo+1))  # ψ_∂σxz∂z
                ]
            ))
            addfield!(grid, "ψ_∂v∂x" => MultiVariableField(
                [
                    backend.zeros(T, 2(halo+1), gridsize[2]  ), # ψ_∂vx∂x
                    backend.zeros(T, 2halo    , gridsize[2]-1)  # ψ_∂vz∂x
                ]
            ))
            addfield!(grid, "ψ_∂v∂z" => MultiVariableField(
                [
                    backend.zeros(T, gridsize[1]-1, 2halo    ), # ψ_∂vx∂z
                    backend.zeros(T, gridsize[1]  , 2(halo+1))  # ψ_∂vz∂z
                ]
            ))
            # Initialize gradient arrays if needed
            if gradient
                # Interpolated material properties
                addfield!(grid, "λ_ihalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [1, 0])...)
                ))
                addfield!(grid, "λ_jhalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [0, 1])...)
                ))
                addfield!(grid, "μ_ihalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [1, 0])...)
                ))
                addfield!(grid, "μ_jhalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [0, 1])...)
                ))
                # Stress and velocity
                addfield!(grid, "adjσ" => MultiVariableField(
                    [
                        backend.zeros(T, gridsize...),              # σxx
                        backend.zeros(T, gridsize...),              # σzz
                        backend.zeros(T, (gridsize .- 1)...)        # σxz
                    ]
                ))
                addfield!(grid, "adjv" => MultiVariableField(
                    [
                        backend.zeros(T, (gridsize .- [1, 0])...),  # vx
                        backend.zeros(T, (gridsize .- [0, 1])...)   # vz
                    ]
                ))
                # CPML memory variables
                    addfield!(grid, "adjψ_∂σ∂x" => MultiVariableField(
                    [
                        backend.zeros(T, 2halo    , gridsize[2]  ), # ψ_∂σxx∂x
                        backend.zeros(T, 2halo    , gridsize[2]  ), # ψ_∂σzz∂x
                        backend.zeros(T, 2(halo+1), gridsize[2]-1)  # ψ_∂σxz∂x
                    ]
                ))
                addfield!(grid, "adjψ_∂σ∂z" => MultiVariableField(
                    [
                        backend.zeros(T, gridsize[1]  , 2halo    ), # ψ_∂σxx∂z
                        backend.zeros(T, gridsize[1]  , 2halo    ), # ψ_∂σzz∂z
                        backend.zeros(T, gridsize[1]-1, 2(halo+1))  # ψ_∂σxz∂z
                    ]
                ))
                addfield!(grid, "adjψ_∂v∂x" => MultiVariableField(
                    [
                        backend.zeros(T, 2(halo+1), gridsize[2]  ), # ψ_∂vx∂x
                        backend.zeros(T, 2halo    , gridsize[2]-1)  # ψ_∂vz∂x
                    ]
                ))
                addfield!(grid, "adjψ_∂v∂z" => MultiVariableField(
                    [
                        backend.zeros(T, gridsize[1]-1, 2halo    ), # ψ_∂vx∂z
                        backend.zeros(T, gridsize[1]  , 2(halo+1))  # ψ_∂vz∂z
                    ]
                ))
                # Gradient arrays
                addfield!(grid, "grad_λ" => ScalarVariableField(
                    backend.zeros(T, gridsize...)
                ))
                addfield!(grid, "grad_μ" => ScalarVariableField(
                    backend.zeros(T, gridsize...)
                ))
                addfield!(grid, "grad_ρ_ihalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [1, 0])...)
                ))
                addfield!(grid, "grad_ρ_jhalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- [0, 1])...)
                ))
                addfield!(grid, "grad_μ_ihalf_jhalf" => ScalarVariableField(
                    backend.zeros(T, (gridsize .- 1)...)
                ))

                # Initialize checkpointer
                checkpointer = LinearCheckpointer(
                    nt,
                    check_freq === nothing ? 1 : check_freq,
                    filter(p -> p.first in ["σ", "v", "ψ_∂σ∂x", "ψ_∂σ∂z", "ψ_∂v∂x", "ψ_∂v∂z"], grid.fields),
                    ["v"];
                    widths=Dict("v" => 1)
                )
                # Save first two timesteps
                savecheckpoint!(checkpointer, "σ" => grid.fields["σ"], 0)
                savecheckpoint!(checkpointer, "v" => grid.fields["v"], 0)
                savecheckpoint!(checkpointer, "ψ_∂σ∂x" => grid.fields["ψ_∂σ∂x"], 0)
                savecheckpoint!(checkpointer, "ψ_∂σ∂z" => grid.fields["ψ_∂σ∂z"], 0)
                savecheckpoint!(checkpointer, "ψ_∂v∂x" => grid.fields["ψ_∂v∂x"], 0)
                savecheckpoint!(checkpointer, "ψ_∂v∂z" => grid.fields["ψ_∂v∂z"], 0)
            end
        else
            error("Only elastic 2D is currently implemented.")
        end

        if snapevery !== nothing
            # Initialize snapshotter
            snapshotter = LinearSnapshotter{Array{T, N}}(nt, snapevery, Dict(
                "v" => MultiVariableField(
                    [
                        backend.zeros(T, (gridsize .- [1, 0])...),  # vx
                        backend.zeros(T, (gridsize .- [0, 1])...)   # vz
                    ]
                ),
                "σ" => MultiVariableField(
                    [
                        backend.zeros(T, gridsize...),              # σxx
                        backend.zeros(T, gridsize...),              # σzz
                        backend.zeros(T, (gridsize .- 1)...)        # σxz
                    ]
                )
            ))
        end

        # Check infoevery
        if infoevery === nothing
            infoevery = nt + 2  # never reach it
        else
            @assert infoevery >= 1 && infoevery <= nt "Infoevery parameter must be positive and less then nt!"
        end

        # Deep copy the material properties
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
            sincinterp,
            snapevery === nothing ? nothing : snapshotter,
            parall
        )
    end
end

###########################################################

# Specific functions for ElasticIsoCPMLWaveSimulation

function reset!(model::ElasticIsoCPMLWaveSimulation{T, N}) where {T, N}
    # Reset computational arrays
    reset!(model.grid; except=["λ", "μ", "ρ_ihalf", "ρ_jhalf", "λ_ihalf", "λ_jhalf", "μ_ihalf", "μ_jhalf", "μ_ihalf_jhalf"])
    if model.checkpointer !== nothing
        reset!(model.checkpointer)
    end
end

init_gradient(model::ElasticIsoCPMLWaveSimulation) = Dict(
    "rho" => zero(model.matprop.ρ),
    "lambda" => zero(model.matprop.λ),
    "mu" => zero(model.matprop.μ),
)

function accumulate_gradient!(totgrad::D, curgrad::D, ::ElasticIsoCPMLWaveSimulation{T, N}) where {T, N, D <: Dict{String, Array{T, N}}}
    totgrad["rho"] .+= curgrad["rho"]
    totgrad["lambda"] .+= curgrad["lambda"]
    totgrad["mu"] .+= curgrad["mu"]
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
