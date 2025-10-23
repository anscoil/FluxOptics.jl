@testset "Modes" begin
    @testset "Gaussian Mode" begin
        w0 = 20.0
        gaussian = Gaussian(w0)

        # Test evaluation at origin
        xv, yv = spatial_vectors(65, 65, 1.0, 1.0)
        field = gaussian(xv, yv)
        center_idx = 33
        @test abs(field[center_idx, center_idx]) == maximum(abs.(field))

        # Test with shift
        shift = Shift2D(5.0, 3.0)
        g_shifted = Gaussian(w0)
        field_shifted = g_shifted(xv, yv, shift)
        # Peak should be at (5, 3)
        @test abs(field_shifted[center_idx, center_idx]) <
              abs(field[center_idx, center_idx])
    end

    @testset "Hermite-Gaussian Modes" begin
        w0 = 15.0
        xv, yv = spatial_vectors(129, 129, 1.0, 1.0)

        # HG₀₀ is Gaussian
        hg00 = HermiteGaussian(w0, 0, 0)
        gaussian = Gaussian(w0)
        field_hg00 = hg00(xv, yv)
        field_gauss = gaussian(xv, yv)
        @test maximum(abs.(field_hg00 - field_gauss)) < 1e-10

        # HG₁₀ has zero at center (odd)
        hg10 = HermiteGaussian(w0, 1, 0)
        field_hg10 = hg10.(xv, yv')
        center_idx = 65
        @test abs(field_hg10[center_idx, center_idx]) < 1e-10

        # HG₂₀ has even symmetry in x
        hg20 = HermiteGaussian(w0, 2, 0)
        field_hg20 = hg20.(xv, yv')
        @test abs(field_hg20[center_idx + 5, center_idx]) ≈
              abs(field_hg20[center_idx - 5, center_idx]) atol = 1e-10
    end

    @testset "Laguerre-Gaussian Modes" begin
        w0 = 20.0
        xv, yv = spatial_vectors(129, 129, 1.0, 1.0)

        # LG₀₀ is Gaussian
        lg00 = LaguerreGaussian(w0, 0, 0)
        gaussian = Gaussian(w0)
        field_lg00 = lg00(xv, yv)
        field_gauss = gaussian(xv, yv)
        @test maximum(abs.(field_lg00 - field_gauss)) < 1e-10

        # LG with l≠0 has vortex (zero at center)
        lg10 = LaguerreGaussian(w0, 0, 1)
        field_lg10 = lg10(xv, yv)
        center_idx = 65
        @test abs(field_lg10[center_idx, center_idx]) < 1e-10
    end

    @testset "HG Mode Groups" begin
        w0 = 12.0
        n_groups = 3
        groups = hermite_gaussian_groups(w0, n_groups)

        # Should have modes up to order 2 (3 groups)
        # Order 0: (0,0) = 1 mode
        # Order 1: (0,1), (1,0) = 2 modes  
        # Order 2: (0,2), (1,1), (2,0) = 3 modes
        # Total: 6 modes
        @test length(groups) == 6
    end

    @testset "Speckle Generation" begin
        ns = (256, 256)
        ds = (1.0, 1.0)
        λ = 1.064
        NA = 0.25

        speckle = generate_speckle(ns, ds, λ, NA)
        @test size(speckle) == ns
        @test eltype(speckle) <: Complex

        # With envelope
        envelope = Gaussian(80.0)
        speckle_env = generate_speckle(ns, ds, λ, NA; envelope = envelope)
        @test size(speckle_env) == ns

        # Edges should be suppressed with envelope
        @test abs(speckle_env[1, 1]) < abs(speckle[1, 1]) || abs(speckle[1, 1]) < 0.1
    end

    @testset "Mode Stack Generation" begin
        ns = (256, 256)
        ds = (1.0, 1.0)
        gaussian = Gaussian(10.0)

        # Grid layout
        layout = Modes.GridLayout(3, 3, 60.0, 60.0)
        mode_stack = generate_mode_stack(layout, ns[1], ns[2], ds[1], ds[2], gaussian)

        @test size(mode_stack) == (ns[1], ns[2], 9)  # 3x3 = 9 modes

        # With different modes
        hg_modes = [HermiteGaussian(12.0, m, n) for m in 0:2 for n in 0:2]
        mode_stack_hg = generate_mode_stack(layout, ns[1], ns[2], ds[1], ds[2], hg_modes)
        @test size(mode_stack_hg) == (ns[1], ns[2], 9)
    end

    @testset "Mode Normalization" begin
        w0 = 20.0
        xv, yv = spatial_vectors(128, 128, 1.0, 1.0)
        ds = (1.0, 1.0)

        gaussian = Gaussian(w0)
        field = gaussian(xv, yv)

        # Calculate power
        power_val = sum(abs2.(field)) * prod(ds)
        @test power_val > 0

        # Normalized field should have unit power
        field_norm = field ./ sqrt(power_val / prod(ds))
        power_norm = sum(abs2.(field_norm)) * prod(ds)
        @test power_norm ≈ 1.0 rtol = 1e-6
    end
end
