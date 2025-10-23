@testset "Metrics" begin
    @testset "Coupling Efficiency" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)

        # Create two Gaussian fields
        u1 = ScalarField(Gaussian(20.0)(x, y), ds, λ)
        u2 = ScalarField(Gaussian(20.0)(x, y), ds, λ)

        normalize_power!(u1, 1.0)
        normalize_power!(u2, 1.0)

        # Perfect overlap
        η = coupling_efficiency(u1, u2)
        @test η[] ≈ 1.0 rtol = 1e-6

        # Orthogonal modes (different sizes)
        u3 = ScalarField(Gaussian(10.0)(x, y), ds, λ)
        normalize_power!(u3, 1.0)
        η_diff = coupling_efficiency(u1, u3)
        @test 0.0 < η_diff[] < 1.0

        # Shifted Gaussian (reduced overlap)
        u4 = ScalarField(Gaussian(20.0)(x .- 30, y), ds, λ)
        normalize_power!(u4, 1.0)
        η_shifted = coupling_efficiency(u1, u4)
        @test η_shifted[] < 0.5  # Significant shift reduces coupling
    end

    @testset "Power Coupling" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        x, y = spatial_vectors(ns, ds)

        u1 = ScalarField(Gaussian(15.0)(x, y), ds, λ)
        u2 = ScalarField(Gaussian(20.0)(x, y), ds, λ)

        # Power coupling
        metric = PowerCoupling(u1)
        P_coupled = metric(u2)
        @test P_coupled[] > 0.92

        # Should be close to min(power(u1), power(u2)) for good overlap
        P1 = power(u1)
        P2 = power(u2)
        @test P_coupled[] ≤ min(P1[], P2[])
    end

    @testset "SquaredIntensityDifference" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0

        u = ScalarField(ones(ComplexF64, ns), ds, λ)
        target = 2.0 .* ones(ns)

        metric = SquaredIntensityDifference((u, target))

        # Compute metric
        result = metric(u)[]

        # Should be sum((|u|² - target)²)
        # |u|² = 1, target = 2, diff = 1, squared = 1
        expected = prod(ns) * 1.0 * prod(ds)
        @test result ≈ expected rtol = 1e-6

        # Test with matching intensities
        u_match = ScalarField(sqrt(2.0) .* ones(ComplexF64, ns), ds, λ)
        result_match = metric(u_match)[]

        @test result_match < 1e-10
    end

    @testset "SquaredFieldDifference" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0

        u = ScalarField(ones(ComplexF64, ns), ds, λ)
        target = ScalarField(2.0 .* ones(ComplexF64, ns), ds, λ)

        metric = SquaredFieldDifference(target)

        # Compute metric
        result = metric(u)[]

        # Should be sum(|u - target|²)
        # |1 - 2|² = 1
        expected = prod(ns) * 1.0 * prod(ds)
        @test result ≈ expected rtol = 1e-6

        # Perfect match
        metric_self = SquaredFieldDifference(u)
        result_self = metric_self(u)[]
        @test result_self ≈ 0.0 atol = 1e-10
    end

    @testset "Metric Backpropagation" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0
        x, y = spatial_vectors(ns, ds)

        # Source field
        u0 = ScalarField(Gaussian(10.0)(x, y), ds, λ)

        # Target
        target_data = Gaussian(10.0)(x .- 5, y)
        target = ScalarField(target_data, ds, λ)

        # System with phase
        source = ScalarSource(u0)
        phase = Phase(u0, zeros(Float32, ns); trainable = true, buffered = true)
        system = source |> phase

        # Metric
        metric = SquaredFieldDifference(target)

        # Loss function
        loss(sys) = sum(metric(sys().out))

        # Should be differentiable
        l, g = Zygote.withgradient(loss, system)

        @test l isa Number
        @test g[1] isa NamedTuple
        @test !isnothing(g[1])
    end

    @testset "Multiple Fields Metric" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0

        u1 = ScalarField(ones(ComplexF64, ns), ds, λ)
        u2 = ScalarField(2.0 .* ones(ComplexF64, ns), ds, λ)

        target1 = ones(ns)
        target2 = 4.0 .* ones(ns)

        # Metric with two fields
        metric = SquaredIntensityDifference((u1, target1), (u2, target2))

        # Should handle multiple field-target pairs
        result = metric(u1, u2)
        @test result isa AbstractArray
        @test length(result) == 2
        @test iszero(result)
    end
end
