@testset "OptimisersExt" begin
    @testset "Proximal Operators - Clamp" begin
        # ClampProx
        clamp_prox = ClampProx(0.0, 1.0)
        x = [-0.5, 0.3, 1.2, 0.8]
        x_copy = copy(x)

        prox_state = ProximalOperators.init(clamp_prox, x_copy)
        ProximalOperators.apply!(clamp_prox, prox_state, x_copy)

        @test x_copy[1] ≈ 0.0
        @test x_copy[2] ≈ 0.3
        @test x_copy[3] ≈ 1.0
        @test x_copy[4] ≈ 0.8
    end

    @testset "Proximal Operators - Positive" begin
        # PositiveProx
        pos_prox = PositiveProx()
        x = [-2.0, -0.5, 0.0, 1.5, 3.0]
        x_copy = copy(x)

        prox_state = ProximalOperators.init(pos_prox, x_copy)
        ProximalOperators.apply!(pos_prox, prox_state, x_copy)

        @test all(x_copy .>= 0.0)
        @test x_copy[1] ≈ 0.0
        @test x_copy[4] ≈ 1.5
    end

    @testset "Proximal Operators - ISTA" begin
        # IstaProx (soft thresholding)
        λ = 0.5
        ista_prox = IstaProx(λ)

        x = [-2.0, -0.3, 0.0, 0.4, 1.5]
        x_copy = copy(x)

        prox_state = ProximalOperators.init(ista_prox, x_copy)
        ProximalOperators.apply!(ista_prox, prox_state, x_copy)

        # Soft thresholding: sign(x) * max(|x| - λ, 0)
        @test x_copy[1] ≈ -1.5  # sign(-2) * max(2 - 0.5, 0)
        @test x_copy[2] ≈ 0.0   # |−0.3| < λ → 0
        @test x_copy[3] ≈ 0.0
        @test x_copy[4] ≈ 0.0   # |0.4| < λ → 0
        @test x_copy[5] ≈ 1.0   # sign(1.5) * max(1.5 - 0.5, 0)
    end

    @testset "Optimization Rules - Fista" begin
        # Simple quadratic optimization: min_x (x - 3)²
        target = 3.0
        x = [0.0]

        loss(x) = (x[1] - target)^2

        opt_state = FluxOptics.setup(Fista(0.9), x)

        for i in 1:50
            l = loss(x)
            g = [2 * (x[1] - target)]  # Gradient
            FluxOptics.update!(opt_state, x, g)
        end

        @test x[1] ≈ target atol = 1e-3
    end

    @testset "Optimization Rules - ProxRule" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Phase mask with ProxRule (clamped optimization)
        phase_data = zeros(Float32, ns)
        phase = Phase(u, phase_data; trainable = true, buffered = true)

        rule = Fista(0.1)
        prox = ClampProx(-π, π)
        prox_rule = ProxRule(rule, prox)

        # Setup should work
        opt_state = FluxOptics.setup(prox_rule, phase)
        @test opt_state isa NamedTuple
    end

    @testset "Optimization - make_rules" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.064
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Two components with different rules
        phase1 = Phase(u, zeros(ns); trainable = true, buffered = true)
        phase2 = Phase(u, zeros(ns); trainable = true, buffered = true)

        rule1 = Fista(0.1)
        rule2 = Fista(0.05)

        rules_dict = make_rules(phase1 => rule1, phase2 => rule2)

        @test haskey(rules_dict, phase1.ϕ)
        @test haskey(rules_dict, phase2.ϕ)
        @test rules_dict[phase1.ϕ] === rule1
        @test rules_dict[phase2.ϕ] === rule2
    end

    @testset "Integrated Optimization Example" begin
        # Simple beam shaping optimization
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        x, y = spatial_vectors(ns, ds)

        # Source
        u0 = ScalarField(Gaussian(15.0)(x, y), ds, λ)
        normalize_power!(u0, 1.0)

        # Target (shifted Gaussian)
        target = Gaussian(15.0)(x .- 10, y)
        target ./= sqrt(sum(abs2.(target)) * prod(ds))

        # System with trainable phase
        source = ScalarSource(u0)
        phase = Phase(u0, zeros(Float32, ns); trainable = true, buffered = true)
        prop = ASProp(u0, 1000.0)
        system = source |> phase |> prop

        # Loss
        loss(sys) = sum(abs2, abs2.(sys().out.electric) - abs2.(target))

        # Optimize
        opt = FluxOptics.setup(Fista(100.0), system)

        initial_loss = loss(system)

        for i in 1:20
            l, g = Zygote.withgradient(loss, system)
            FluxOptics.update!(opt, system, g[1])
        end

        final_loss = loss(system)

        # Loss should decrease
        @test final_loss < initial_loss
    end
end
