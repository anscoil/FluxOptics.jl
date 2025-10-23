@testset "GridUtils" begin
    @testset "Spatial Vectors - 1D" begin
        # Single dimension
        x, = spatial_vectors(64, 2.0)
        @test length(x) == 64
        @test step(x) ≈ 2.0
        # Center at zero by default
        @test x[32] + x[33] ≈ 0.0 atol = 1e-10

        # With offset
        x_offset, = spatial_vectors(64, 2.0; xc = 5.0)
        @test mean(x_offset) ≈ -5.0 rtol = 1e-10
    end

    @testset "Spatial Vectors - 2D" begin
        # Basic 2D grid
        xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
        @test length(xv) == 128
        @test length(yv) == 128
        @test step(xv) ≈ 2.0
        @test step(yv) ≈ 2.0

        # With offset
        xv_off, yv_off = spatial_vectors(128, 128, 2.0, 2.0; xc = -5.0, yc = 30.5)
        @test mean(xv_off) ≈ 5.0 rtol = 1e-10
        @test mean(yv_off) ≈ -30.5 rtol = 1e-10

        # Tuple version
        xv2, yv2 = spatial_vectors((128, 128), (2.0, 2.0); offset = (-5.0, 30.5))
        @test mean(xv2) ≈ 5.0 rtol = 1e-10
        @test mean(yv2) ≈ -30.5 rtol = 1e-10
    end

    @testset "Coordinate Transformations - Shift2D" begin
        # Basic shift
        shift = Shift2D(10.0, -5.0)
        point = [0.0, 0.0]
        shifted = shift(point)
        @test shifted[1] ≈ 10.0
        @test shifted[2] ≈ -5.0

        # Apply to array of points
        shifted2 = shift([5.0, 3.0])
        @test shifted2[1] ≈ 15.0
        @test shifted2[2] ≈ -2.0
    end

    @testset "Coordinate Transformations - Rot2D" begin
        # 90 degree rotation
        rot90 = Rot2D(π / 2)
        point = [1.0, 0.0]
        rotated = rot90(point)
        @test rotated[1] ≈ 0.0 atol = 1e-10
        @test rotated[2] ≈ 1.0 atol = 1e-10

        # 45 degree rotation
        rot45 = Rot2D(π / 4)
        point2 = [1.0, 0.0]
        rotated2 = rot45(point2)
        @test rotated2[1] ≈ 1 / √2 atol = 1e-10
        @test rotated2[2] ≈ 1 / √2 atol = 1e-10
    end

    @testset "Coordinate Transformations - Composition" begin
        # Shift then rotate
        transform = Rot2D(π / 4) ∘ Shift2D(10.0, 5.0)
        point = [0.0, 0.0]
        result = transform(point)

        # Should first shift to (10, 5), then rotate
        expected_x = (10 - 5) / √2
        expected_y = (10 + 5) / √2
        @test result[1] ≈ expected_x atol = 1e-10
        @test result[2] ≈ expected_y atol = 1e-10
    end

    @testset "Id2D Transformation" begin
        identity_transform = Id2D()
        point = [5.0, 3.0]
        result = identity_transform(point)
        @test result[1] == 5.0
        @test result[2] == 3.0
    end
end
