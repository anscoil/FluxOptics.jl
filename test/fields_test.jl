@testset "Fields" begin
    @testset "ScalarField Construction" begin
        # Basic construction
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        data = ones(ComplexF64, ns)

        u = ScalarField(data, ds, λ)
        @test size(u) == ns
        @test Tuple(u.ds) == ds
        @test u.lambdas.val == u.lambdas.collection == λ

        # With tilts
        tilts = (0.1, 0.05)
        u_tilt = ScalarField(data, ds, λ; tilts = tilts)
        @test u_tilt.tilts.collection == map(x -> [x], tilts)
    end

    @testset "Multi-wavelength Field" begin
        ns = (64, 64, 3)
        ds = (1.0, 1.0)
        λs = [0.640, 0.538, 0.455]
        data = ones(ComplexF64, ns)

        u = ScalarField(data, ds, λs)
        @test size(u) == ns
        @test length(u.lambdas.collection) == 3
        @test u.lambdas.collection[1] == 0.640
        @test u.lambdas.collection[3] == 0.455
    end

    @testset "Field Indexing and Access" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0
        data = reshape(1:(32 * 32), ns) |> collect |> complex

        u = ScalarField(data, ds, λ)

        # Test indexing
        @test u.electric[1, 1] == complex(1.0)
        @test u.electric[32, 32] == complex(32*32)

        # Test size
        @test size(u) == ns
    end

    @testset "Field Power and Normalization" begin
        ns = (128, 128)
        ds = (0.5, 0.5)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)

        w0 = 20.0
        data = Gaussian(w0)(x, y)
        u = ScalarField(data, ds, λ)

        # Compute power
        P_initial = power(u)
        @test all((>)(0), P_initial)

        # Normalize to unit power
        normalize_power!(u, 1.0)
        P_normalized = power(u)
        @test all(x -> isapprox(x, 1; rtol = 1e-6), P_normalized)

        # Normalize to different power
        P_target = 2.5
        normalize_power!(u, P_target)
        @test all(x -> isapprox(x, P_target; rtol = 1e-6), power(u))
    end

    @testset "Field Arithmetic" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0

        u1 = ScalarField(ones(ComplexF64, ns), ds, λ)
        u2 = ScalarField(2 .* ones(ComplexF64, ns), ds, λ)

        # Addition
        u_sum = u1 + u2
        @test all(u_sum.electric .≈ 3.0)

        # Subtraction
        u_diff = u2 - u1
        @test all(u_diff.electric .≈ 1.0)

        # Scalar multiplication
        u_scaled = 3 * u1
        @test all(u_scaled.electric .≈ 3.0)
    end

    @testset "Field Copy and Mutation" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0
        data = ones(ComplexF64, ns)

        u1 = ScalarField(data, ds, λ)
        u2 = copy(u1)

        # Modify u2
        u2.electric[1, 1] = 2.0 + 0im

        # Check u1 unchanged
        @test u1.electric[1, 1] == 1.0 + 0im
        @test u2.electric[1, 1] == 2.0 + 0im
    end

    @testset "Field Properties" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)

        # Create field with phase structure
        data = exp.(1im .* (x .+ y'))
        u = ScalarField(data, ds, λ)

        # Test intensity
        I = intensity(u)
        @test all(I .≈ 1.0)  # Constant amplitude

        # Test phase
        ϕ = phase(u)
        @test size(ϕ) == ns
        @test all(-π .<= ϕ .<= π)
    end
end
