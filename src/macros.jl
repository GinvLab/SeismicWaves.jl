


macro threadpersource(expr)
    return esc(quote
                   if parapersrc == :threadpersrc
                       Threads.@threads $(expr)
                   else
                       $(expr)
                   end
               end)
end
