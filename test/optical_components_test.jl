@testset "OpticalComponents" begin
    @testset "Sources" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)
        data = Gaussian(20.0)(x, y)
        u0 = ScalarField(data, ds, λ)

        # ScalarSource
        source = ScalarSource(u0)
        @test !istrainable(source)

        result = propagate(source)
        @test result.electric ≈ u0.electric

        # Trainable source
        source_train = ScalarSource(u0; trainable = true)
        @test istrainable(source_train)
    end

    @testset "Modulators - Phase" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Constant phase
        phase_const = Phase(u, (x, y) -> π / 2)
        result = propagate(u, phase_const, Forward)
        @test all(angle.(result.electric) .≈ π / 2)

        # Spatially varying phase
        x, y = spatial_vectors(ns, ds)
        phase_func = Phase(u, (x, y) -> x + y)
        result_func = propagate(u, phase_func, Forward)
        expected_phase = x .+ y'
        @test all(angle.(result_func.electric) .≈ angle.(cis.(expected_phase)))

        # Trainable phase mask
        phase_data = zeros(Float32, ns)
        phase_train = Phase(u, phase_data; trainable = true, buffered = true)
        @test istrainable(phase_train)
    end

    @testset "Modulators - Mask" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Circular aperture
        x, y = spatial_vectors(ns, ds)
        R = 20.0
        aperture = Mask(u, (x, y) -> x^2 + y^2 < R^2 ? 1.0 : 0.0)

        result = propagate(u, aperture, Forward)

        # Check center is transmitted
        center_idx = ns[1] ÷ 2
        @test abs(result.electric[center_idx, center_idx]) ≈ 1.0

        # Check edges are blocked
        @test abs(result.electric[1, 1]) < 0.1
    end

    @testset "Modulators - TeaDOE" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Refractive index function
        dn = λ -> 0.5

        # TeaDOE with zero height
        doe = TeaDOE(u, dn; trainable = true, buffered = true)
        @test istrainable(doe)

        result = propagate(u, doe, Forward)
        # Zero height should give identity
        @test result.electric ≈ u.electric
    end

    @testset "Propagators - Unitarity" begin
        ns = (128, 128)
        ds = (1.0, 1.0)
        λ = 1.064
        z = 1000.0
        x, y = spatial_vectors(ns, ds)

        # Create Gaussian field
        data = Gaussian(20.0)(x, y)
        u_forward = ScalarField(data, ds, λ)

        # Test Angular Spectrum
        prop_as = ASProp(u_forward, z)
        u_prop = propagate(u_forward, prop_as, Forward)
        u_back = propagate(u_prop, prop_as, Backward)

        # Check unitarity (forward-backward ≈ identity)
        coupling = coupling_efficiency(u_forward, u_back)[]
        @test coupling > 0.9999  # Very high coupling for unitarity

        # Test Rayleigh-Sommerfeld
        prop_rs = RSProp(u_forward, z)
        u_prop_rs = propagate(u_forward, prop_rs, Forward)
        u_back_rs = propagate(u_prop_rs, prop_rs, Backward)

        coupling_rs = coupling_efficiency(u_forward, u_back_rs)[]
        @test coupling_rs > 0.999  # High coupling
    end

    @testset "Propagators - Energy Conservation" begin
        ns = (128, 128)
        ds = (0.5, 0.5)
        λ = 1.064
        z = 500.0
        x, y = spatial_vectors(ns, ds)

        data = Gaussian(15.0)(x, y)
        u = ScalarField(data, ds, λ)
        normalize_power!(u, 1.0)

        # Angular Spectrum propagation
        prop = ASProp(u, z)
        u_prop = propagate(u, prop, Forward)

        # Power should be conserved
        P_before = power(u)
        P_after = power(u_prop)
        @test P_after ≈ P_before rtol = 1e-6
    end

    @testset "Propagators - Paraxial" begin
        ns = (64, 64)
        ds = (2.0, 2.0)
        λ = 1.064
        z = 1000.0
        x, y = spatial_vectors(ns, ds)

        data = Gaussian(30.0)(x, y)
        u = ScalarField(data, ds, λ)

        # Paraxial propagation
        prop_paraxial = ParaxialProp(u, ds, z)
        u_prop = propagate(u, prop_paraxial, Forward)

        @test size(u_prop) == size(u)
        @test power(u_prop)[] > 0
    end

    @testset "Propagators - Collins Integral" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        x, y = spatial_vectors(ns, ds)

        data = Gaussian(10.0)(x, y)
        u = ScalarField(data, ds, λ)

        # Free space propagation (A=D=1, B=z, C=0)
        z = 100.0
        A, B, D = 1.0, z, 1.0
        prop_collins = CollinsProp(u, ds, (A, B, D))

        u_prop = propagate(u, prop_collins, Forward)
        @test size(u_prop) == size(u)
    end

    @testset "Utilities - Pad and Crop" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        u = ones(ComplexF64, ns)

        # Pad
        ns_padded = (64, 64)
        u_padded = pad(u, ns_padded)
        @test size(u_padded) == ns_padded

        # Crop back
        u_cropped = crop(u_padded, ns)
        @test size(u_cropped) == ns

        # Values should match in center
        offset = (ns_padded .- ns) .÷ 2
        center_val = u_padded[offset[1] + 1, offset[2] + 1]
        @test center_val ≈ 1.0
    end

    @testset "System Composition" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)
        data = Gaussian(15.0)(x, y)
        u0 = ScalarField(data, ds, λ)

        # Build system with pipe
        source = ScalarSource(u0)
        phase_mask = Phase(u0, (x, y) -> π / 4)
        prop = ASProp(u0, 500.0)

        system = source |> phase_mask |> prop

        # Execute system
        result = system()
        @test result.out isa ScalarField
        @test size(result.out) == ns
    end

    @testset "Field Probes" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.0
        u0 = ScalarField(ones(ComplexF64, ns), ds, λ)

        source = ScalarSource(u0)
        probe1 = FieldProbe()
        phase = Phase(u0, (x, y) -> π / 2)
        probe2 = FieldProbe()

        system = source |> probe1 |> phase |> probe2

        result = system()

        # Check probes captured fields
        @test haskey(result.probes, probe1)
        @test haskey(result.probes, probe2)

        # Probe 1 should have original field
        @test result.probes[probe1].electric ≈ u0.electric

        # Probe 2 should have phase-shifted field
        @test all(angle.(result.probes[probe2].electric) .≈ π / 2)
    end

    @testset "Trainability" begin
        ns = (32, 32)
        ds = (1.0, 1.0)
        λ = 1.0
        u = ScalarField(ones(ComplexF64, ns), ds, λ)

        # Static component
        phase_static = Phase(u, (x, y) -> 0.0)
        @test !istrainable(phase_static)
        @test !isbuffered(phase_static)

        # Trainable component
        phase_train = Phase(u, zeros(ns); trainable = true)
        @test istrainable(phase_train)
        @test !isbuffered(phase_train)

        # Buffered component
        phase_buffered = Phase(u, zeros(ns); trainable = true, buffered = true)
        @test istrainable(phase_buffered)
        @test isbuffered(phase_buffered)
    end

    @testset "BPM Propagation" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        x, y = spatial_vectors(ns, ds)

        data = Gaussian(10.0)(x, y)
        u0 = ScalarField(data, ds, λ)

        # Define refractive index profile
        n_bulk = 1.5
        Δn = zeros(ns..., 20)  # 20 propagation steps
        L = 100.0  # Total length

        bpm = AS_BPM(u0, L, n_bulk, Δn)
        result = propagate(u0, bpm, Forward)

        @test result isa ScalarField
        @test size(result)[1:2] == ns
    end

    @testset "Active Media - Gain" begin
        ns = (64, 64)
        ds = (1.0, 1.0)
        λ = 1.064
        dz = 1000.0
        Isat = 1.0
        x, y = spatial_vectors(ns, ds)

        data = Gaussian(15.0)(x, y)
        u = ScalarField(data, ds, λ)
        normalize_power!(u, 0.1)

        # Uniform gain
        g0 = 0.01
        gain = GainSheet(u, dz, Isat, (x, y) -> g0)

        result = propagate(u, gain, Forward)

        # Power should increase due to gain
        P_before = power(u)[]
        P_after = power(result)[]
        @test P_after > P_before
    end
end
