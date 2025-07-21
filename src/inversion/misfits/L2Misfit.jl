
struct L2Misfit <: AbstractMisfit
    observed::Array#{T,N}
    invcov::AbstractMatrix#{T}
    windows::Vector{Pair{Int, Int}}

    function L2Misfit(; observed::Union{Array, Nothing}=nothing,
                          invcov::Union{AbstractMatrix, Nothing}=nothing,
                          windows::Vector{Pair{Int, Int}}=Vector{Pair{Int, Int}}()
                          )
        
        return new(observed,invcov,windows)
    end
end



function calcmisfit(shotmisfit::L2Misfit,recs::ScalarReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in  shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = dot(residuals,  shotmisfit.invcov, residuals) / 2
    # # Add regularization if needed
    # if misfit.regularization !== nothing
    #     msf += misfit.regularization(matprop)
    # end
    return msf
end

function calcmisfit(shotmisfit::L2Misfit,recs::VectorReceivers{T, N}) where {T, N}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Inner product using inverse covariance matrix
    msf = sum([dot(residuals[:, i, :], shotmisfit.invcov, residuals[:, i, :]) for i in 1:N]) / 2
    # Add regularization if needed
    # if misfit.regularization !== nothing
    #     msf += misfit.regularization(matprop)
    # end
    return msf
end

function ∂χ_∂u(shotmisfit::L2Misfit, recs::ScalarReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    return shotmisfit.invcov * residuals
end

function ∂χ_∂u(shotmisfit::L2Misfit, recs::VectorReceivers{T}) where {T}
    # Compute residuals
    residuals = recs.seismograms - shotmisfit.observed
    # Window residuals using mask
    mask = ones(T, size(residuals, 1))
    if length(shotmisfit.windows) > 0
        for wnd in shotmisfit.windows
            mask[wnd.first:wnd.second] .= 2
        end
        mask .-= 1
    end
    residuals .= mask .* residuals
    # Multiply with inverse of covariance matrix
    for d in axes(residuals, 2)
        residuals[:, d, :] .= shotmisfit.invcov * residuals[:, d, :]
    end
    return residuals
end
