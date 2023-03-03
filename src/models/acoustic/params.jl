"""
  Parameters for acoustic wave simulations
"""
abstract type InputParametersAcoustic <: InputParameters end

Base.@kwdef struct InputParametersAcoustic1D <: InputParametersAcoustic
    ntimesteps::Int64
    nx::Int64
    dt::Float64
    dh::Float64
    savesnapshot::Bool
    snapevery::Int64
    boundcond::String
    infoevery::Int64
end

Base.@kwdef struct InputParametersAcoustic2D <: InputParametersAcoustic
    ntimesteps::Int64
    nx::Int64
    nz::Int64
    dt::Float64
    dh::Float64
    savesnapshot::Bool
    snapevery::Int64
    boundcond::String
    freeboundtop::Bool
    infoevery::Int64
end

Base.@kwdef struct InputParametersAcoustic3D <: InputParametersAcoustic
    ntimesteps::Int64
    nx::Int64
    ny::Int64
    nz::Int64
    dt::Float64
    dh::Float64
    savesnapshot::Bool
    snapevery::Int64
    boundcond::String
    freeboundtop::Bool
    infoevery::Int64
end