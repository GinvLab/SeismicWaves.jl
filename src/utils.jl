 
##==============================================
"""
 Gaussian source time function
"""
function gaussource1D( t::Vector{<:Real}, t0::Real, f0::Real )
    # boh = f0 .* (t-t0)
    # source = -8.0*boh.*exp( -boh.^2/(4.0*f0)^2 )
    boh= pi.*f0.*(t.-t0)
    source = -boh.*exp.( -boh.^2 )    
    return source
end

##========================================
"""
 Ricker source time function
"""
function rickersource1D( t::Vector{<:Real},  t0::Real, f0::Real )    
    b = (pi*f0*(t.-t0)).^2
    w = (1.0.-2.0.*b).*exp.(.-b)
    return w
end

##========================================
