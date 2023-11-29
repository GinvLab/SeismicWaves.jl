macro d_dx_4th(a, i)
    return esc( :( ( -$a[$i+1] + 27.0 * $a[$i] - 27.0 * $a[$i-1] + $a[$i-2] ) ) )
end