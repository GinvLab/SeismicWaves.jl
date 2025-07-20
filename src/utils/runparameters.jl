

struct RunParameters
    parall::Symbol
    snapevery::Union{Int, Nothing}
    infoevery::Union{Int, Nothing}
    logger::AbstractLogger
    erroronCFL::Bool
    erroronPPW::Bool

    function RunParameters(;
                           parall::Symbol=:threads,
                           snapevery::Union{Int, Nothing}=nothing,
                           infoevery::Union{Int, Nothing}=nothing,
                           logger::AbstractLogger=current_logger(),
                           erroronCFL::Bool=true,
                           erroronPPW::Bool=false
                           )
        return new(parall,snapevery,infoevery,logger,erroronCFL,erroronPPW)
    end
end
