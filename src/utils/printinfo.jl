

function printinfoiter(ter::REPL.Terminals.TTYTerminal,it::Integer,nt::Integer,
                       infoevery::Union{Integer,Nothing},dt::Real,kind::Symbol)
    
    if kind==:forw || kind==:adjforw
        pit = it
    elseif kind==:adjback
        pit = nt-it+1
    end

    # Check that the loglevel is <= Info before printing anything
    belowinfolev = Logging.min_enabled_level(current_logger()) <= LogLevel(Info)
    
    if infoevery!=nothing  && belowinfolev  && (pit % infoevery == 0 || pit==1) 
        if pit!=1
            REPL.Terminals.clear_line(ter)
            REPL.Terminals.cmove_line_up(ter)
        end
        if kind==:forw
            @info @sprintf( "Iteration: %d of %d, simulation time: %g s",it, nt, dt*(it-1) )
        elseif kind==:adjforw
            @info @sprintf( "Forward loop: %d of %d, simulation time: %g s",it, nt, dt*(it-1) )
        elseif kind==:adjback
            @info @sprintf( "Adjoint loop: %d of %d, simulation time: %g s",pit, nt, dt*(it-1) )
        end        
    end
    return
end

