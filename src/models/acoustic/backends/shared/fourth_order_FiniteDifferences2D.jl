macro d_dx_4th(a, i, j)
    return esc( :( ( -$a[$i+1, $j] + 27.0 * $a[$i, $j] - 27.0 * $a[$i-1, $j] + $a[$i-2, $j] ) ) )
end

macro d_dy_4th(a, i, j)
    return esc( :( ( -$a[$i, $j+1] + 27.0 * $a[$i, $j] - 27.0 * $a[$i, $j-1] + $a[$i, $j-2] ) ) )
end
