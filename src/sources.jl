"""
$(TYPEDEF)

Type representing a multi-source configuration for a wave propagation shot.

$(TYPEDFIELDS)
"""
struct ScalarSources{T <: Real} <: Sources
    "Source positions"
    positions::Matrix{<:Real}
    "Source time function"
    tf::Matrix{T}
    "Dominant frequency"
    domfreq::T
    
    @doc """ 
        ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function ScalarSources{T}(positions::Matrix{<:Real}, tf::Matrix{T}, domfreq::T) where {T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        return new(positions, tf, domfreq)
    end
end

# Default type constructor {Float64}
@doc """
$(SIGNATURES)

Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
Default type constructor for Float64.
"""
ScalarSources(positions, tf, domfreq) = ScalarSources{Float64}(positions, tf, domfreq)

####################################################

"""
Type representing a 2D moment tensor.
"""
struct MomentTensor2D{T <: Real} <: MomentTensor
  Mxx::Vector{T}
  Mzz::Vector{T}
  Mxz::Vector{T}
end

"""
Type representing a 2D moment tensor.
"""
struct MomentTensor3D{T <: Real} <: MomentTensor
  Mxx::Vector{T}
  Myy::Vector{T}
  Mzz::Vector{T}
  Mxy::Vector{T}
  Mxz::Vector{T}
  Myz::Vector{T}
end


"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct MomentTensorSources{N, T <: Real} <: Sources
    positions::Matrix{<:Real}
    tf::Matrix{T}
    momtens::MomentTensor
    domfreq::T

    @doc """
        MomentTensorSources{T<:Real}( 
            positions::Matrix{<:Real},
            tf::Matrix{<:T},
            momtens::MomentTensor2D
            domfreq::T
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a dominant frequency.
    """
    function MomentTensorSources{N,T}(positions::Matrix{<:Real}, tf::Matrix{T},  momtens::MomentTensor, domfreq::T) where {N, T <: Real}
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == size(tf, 2) "Number of sources do not match between positions and time-functions!"
        if N==1
            @assert typeof(momtens)==MomentTensor2D
        elseif N==2
            @assert typeof(momtens)==MomentTensor3D
        else
            error("MomentTensorSources: Moment tensor neither 2D nor 3D.")
        end
        return new{N,T}(positions, tf, momtens, domfreq)
    end
end

# Default type constructor {Float64}
MomentTensorSources(positions, tf, momtens, domfreq) = MomentTensorSources{Float64}(positions, tf, momtens, domfreq)

####################################################
