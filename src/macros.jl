


macro athreadpersrcornot(expr)
    return esc(quote
                   if parall == :athreadpersrc
                       #println("One thread per source, nthreads: $(Threads.nthreads())")
                       Threads.@threads $(expr)
                   else
                       $(expr)
                   end
               end)
end
