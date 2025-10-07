
using SeismicWaves: interpolate, interp, ArithmeticAverageInterpolation, HarmonicAverageInterpolation
using SeismicWaves: ∂f∂m, back_interp

# Test ArithmeticAverageInterpolation for 1D array
function test_arithmetic_average_interpolation_1d()
    m = [1.0, 2.0, 3.0, 4.0]
    itp = ArithmeticAverageInterpolation()
    @test interpolate(m, itp)[1] == [1.5, 2.5, 3.5]
    @test interp(itp, m, 1) == [1.5, 2.5, 3.5]
    @test ∂f∂m(itp, m, [1, 2, 3], [1]) == [0.5, 0.5, 0.5]
    @test ∂f∂m(itp, m, [2, 3, 4], [1]) == [0.5, 0.5, 0.5]
end

# Test HarmonicAverageInterpolation for 1D array
function test_harmonic_average_interpolation_1d()
    m = [1.0, 2.0, 3.0, 4.0]
    dims = [1]
    itp = HarmonicAverageInterpolation()

    g = [1/m[1] + 1/m[2], 1/m[2] + 1/m[3], 1/m[3] + 1/m[4]]
    
    @test interpolate(m, itp)[1] ≈ [1.333, 2.4, 3.428] atol=1e-3
    @test interp(itp, m, 1) ≈ [1.333, 2.4, 3.428] atol=1e-3
    @test ∂f∂m(itp, m, [1, 2, 3], dims) ≈ (-2 ./ (g.^2)) .* (-1 ./ (m[1:end-1].^2)) atol=1e-3
    @test ∂f∂m(itp, m, [2, 3, 4], dims) ≈ (-2 ./ (g.^2)) .* (-1 ./ (m[2:end  ].^2)) atol=1e-3
end

# Test back_interp for 1D array
function test_back_interp_1d()
    m = [1.0, 2.0, 3.0, 4.0]
    ∂χ∂m_interp = [0.1, 0.2, 0.3]
    dims = [1]
    itp = ArithmeticAverageInterpolation()
    expected = zeros(4)
    expected[1:end-1] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, 1, dims)
    expected[2:end] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, 2, dims)
    
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected
end

# Test back_interp for HarmonicAverageInterpolation for 1D array
function test_back_interp_harmonic_1d()
    m = [1.0, 2.0, 3.0, 4.0]
    ∂χ∂m_interp = [0.1, 0.2, 0.3]
    dims = [1]
    itp = HarmonicAverageInterpolation()
    expected = zeros(4)
    expected[1:end-1] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, [1,2,3], dims)
    expected[2:end] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, [2,3,4], dims)
    
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected
end

# Test ArithmeticAverageInterpolation for 2D array
function test_arithmetic_average_interpolation_2d()
    m = [1.0 2.0 3.0;
         4.0 5.0 6.0;
         7.0 8.0 9.0]
    itp = ArithmeticAverageInterpolation()

    @test interpolate(m, itp)[1] == [
        2.5 3.5 4.5;
        5.5 6.5 7.5
    ]
    @test interpolate(m, itp)[2] == [
        1.5 2.5;
        4.5 5.5;
        7.5 8.5
    ]
    @test interp(itp, m, [1, 2]) == [
        3.0 4.0;
        6.0 7.0
    ]
    @test ∂f∂m(itp, m, [1:2, 1:2], [1, 2]) == [
        0.25 0.25;
        0.25 0.25
    ]
    @test ∂f∂m(itp, m, [1:2, 2:3], [1, 2]) == [
        0.25 0.25;
        0.25 0.25
    ]
    @test ∂f∂m(itp, m, [2:3, 1:2], [1, 2]) == [
        0.25 0.25;
        0.25 0.25
    ]
    @test ∂f∂m(itp, m, [2:3, 2:3], [1, 2]) == [
        0.25 0.25;
        0.25 0.25
    ]
end

# Test HarmonicAverageInterpolation for 2D array
function test_harmonic_average_interpolation_2d()
    m = [1.0 2.0 3.0;
         4.0 5.0 6.0;
         7.0 8.0 9.0]
    itp = HarmonicAverageInterpolation()

    g1 = 1/m[1,1] + 1/m[1,2] + 1/m[2,1] + 1/m[2,2]
    g2 = 1/m[1,2] + 1/m[1,3] + 1/m[2,2] + 1/m[2,3]
    g3 = 1/m[2,1] + 1/m[2,2] + 1/m[3,1] + 1/m[3,2]
    g4 = 1/m[2,2] + 1/m[2,3] + 1/m[3,2] + 1/m[3,3]
    g = [g1 g2; g3 g4]

    @test interpolate(m, itp)[1] ≈ [
        2/(1/m[1,1] + 1/m[2,1]) 2/(1/m[1,2] + 1/m[2,2]) 2/(1/m[1,3] + 1/m[2,3]);
        2/(1/m[2,1] + 1/m[3,1]) 2/(1/m[2,2] + 1/m[3,2]) 2/(1/m[2,3] + 1/m[3,3])
    ]
    @test interpolate(m, itp)[2] ≈ [
        2/(1/m[1,1] + 1/m[1,2]) 2/(1/m[1,2] + 1/m[1,3]);
        2/(1/m[2,1] + 1/m[2,2]) 2/(1/m[2,2] + 1/m[2,3]);
        2/(1/m[3,1] + 1/m[3,2]) 2/(1/m[3,2] + 1/m[3,3])
    ]
    @test interp(itp, m, [1,2]) ≈ 4 ./ g
    @test ∂f∂m(itp, m, CartesianIndices((1:2, 1:2)), [1, 2]) ≈ (-4 ./ g.^2) .* (-1 ./ (m[1:2,1:2].^2))
    @test ∂f∂m(itp, m, CartesianIndices((1:2, 2:3)), [1, 2]) ≈ (-4 ./ g.^2) .* (-1 ./ (m[1:2,2:3].^2))
    @test ∂f∂m(itp, m, CartesianIndices((2:3, 1:2)), [1, 2]) ≈ (-4 ./ g.^2) .* (-1 ./ (m[2:3,1:2].^2))
    @test ∂f∂m(itp, m, CartesianIndices((2:3, 2:3)), [1, 2]) ≈ (-4 ./ g.^2) .* (-1 ./ (m[2:3,2:3].^2))

    g151 = 1/m[1,1] + 1/m[2,1]
    g152 = 1/m[1,2] + 1/m[2,2]
    g153 = 1/m[1,3] + 1/m[2,3]
    g251 = 1/m[2,1] + 1/m[3,1]
    g252 = 1/m[2,2] + 1/m[3,2]
    g253 = 1/m[2,3] + 1/m[3,3]
    g15 = [g151 g152 g153; g251 g252 g253]
    @test interp(itp, m, 1) ≈ [
        2/(1/m[1,1] + 1/m[2,1]) 2/(1/m[1,2] + 1/m[2,2]) 2/(1/m[1,3] + 1/m[2,3]);
        2/(1/m[2,1] + 1/m[3,1]) 2/(1/m[2,2] + 1/m[3,2]) 2/(1/m[2,3] + 1/m[3,3])
    ]
    @test ∂f∂m(itp, m, CartesianIndices((1:2, 1:3)), [1]) ≈ (-2 ./ (g15.^2)) .* (-1 ./ (m[1:2,1:3].^2))
    @test ∂f∂m(itp, m, CartesianIndices((2:3, 1:3)), [1]) ≈ (-2 ./ (g15.^2)) .* (-1 ./ (m[2:3,1:3].^2))

    g115 = 1/m[1,1] + 1/m[1,2]
    g125 = 1/m[1,2] + 1/m[1,3]
    g215 = 1/m[2,1] + 1/m[2,2]
    g225 = 1/m[2,2] + 1/m[2,3]
    g315 = 1/m[3,1] + 1/m[3,2]
    g325 = 1/m[3,2] + 1/m[3,3]
    g25 = [g115 g125; g215 g225; g315 g325]
    @test interp(itp, m, 2) ≈ [
        2/(1/m[1,1] + 1/m[1,2]) 2/(1/m[1,2] + 1/m[1,3]);
        2/(1/m[2,1] + 1/m[2,2]) 2/(1/m[2,2] + 1/m[2,3]);
        2/(1/m[3,1] + 1/m[3,2]) 2/(1/m[3,2] + 1/m[3,3])
    ]
    @test ∂f∂m(itp, m, CartesianIndices((1:3, 1:2)), [2]) ≈ (-2 ./ (g25.^2)) .* (-1 ./ (m[1:3,1:2].^2))
    @test ∂f∂m(itp, m, CartesianIndices((1:3, 2:3)), [2]) ≈ (-2 ./ (g25.^2)) .* (-1 ./ (m[1:3,2:3].^2))
end

# Test back_interp for 2D array
function test_back_interp_2d()
    m = [1.0 2.0 3.0;
         4.0 5.0 6.0;
         7.0 8.0 9.0]
    ∂χ∂m_interp = [0.1 0.2; 0.3 0.4]
    dims = [1, 2]
    itp = ArithmeticAverageInterpolation()
    expected = zeros(3, 3)
    expected[1:2, 1:2] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:2, 1:2)), dims)
    expected[1:2, 2:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:2, 2:3)), dims)
    expected[2:3, 1:2] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((2:3, 1:2)), dims)
    expected[2:3, 2:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((2:3, 2:3)), dims)
    
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected
end

# Test back_interp for HarmonicAverageInterpolation for 2D array
function test_back_interp_harmonic_2d()
    m = [1.0 2.0 3.0;
         4.0 5.0 6.0;
         7.0 8.0 9.0]
    ∂χ∂m_interp = [0.1 0.2; 0.3 0.4]
    dims = [1, 2]
    itp = HarmonicAverageInterpolation()
    expected = zeros(3, 3)
    expected[1:2, 1:2] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:2, 1:2)), dims)
    expected[1:2, 2:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:2, 2:3)), dims)
    expected[2:3, 1:2] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((2:3, 1:2)), dims)
    expected[2:3, 2:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((2:3, 2:3)), dims)
    
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected

    dims = [1]
    expected = zeros(3, 3)
    ∂χ∂m_interp = [0.1 0.2 0.3; 0.4 0.5 0.6]
    expected[1:2, 1:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:2, 1:3)), dims)
    expected[2:3, 1:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((2:3, 1:3)), dims)
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected

    dims = [2]
    expected = zeros(3, 3)
    ∂χ∂m_interp = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    expected[1:3, 1:2] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:3, 1:2)), dims)
    expected[1:3, 2:3] .+= ∂χ∂m_interp .* ∂f∂m(itp, m, CartesianIndices((1:3, 2:3)), dims)
    @test back_interp(itp, m, ∂χ∂m_interp, dims) ≈ expected

end

# Run tests
@testset "Interpolation Tests" begin
    test_arithmetic_average_interpolation_1d()
    test_harmonic_average_interpolation_1d()
    test_back_interp_1d()
    test_back_interp_harmonic_1d()
    test_arithmetic_average_interpolation_2d()
    test_harmonic_average_interpolation_2d()
    test_back_interp_2d()
    test_back_interp_harmonic_2d()
end
