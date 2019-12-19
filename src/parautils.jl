
##==============================================

function distribwork(nelem::Integer,nwks::Integer)
    ## calculate how to subdivide the srcs among the workers
    if nelem>=nwks
        dis = div(nelem,nwks)
        grpsizes = dis*ones(Int64,nwks)        
        resto = mod(nelem,nwks)
        if resto>0
            ## add the reminder 
            grpsizes[1:resto] .+= 1
        end
    else
        ## if more workers than sources use only necessary workers
        grpsizes = ones(Int64,nelem)        
    end
    ## now set the indices for groups of srcs
    grpsrc = zeros(Int64,length(grpsizes),2)
    grpsrc[:,1] = cumsum(grpsizes).-grpsizes.+1
    grpsrc[:,2] = cumsum(grpsizes)
    return grpsrc
end

##==============================================

