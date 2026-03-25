using FluxOptics
using NPZ

data_dir = @__DIR__

ns = (512, 512)
ds = (1.0, 1.0)
λ = 1.064  # in µm
z = 2000.0  # in µm
xv, yv = spatial_vectors(ns, ds)

w_in = 25.0
n_order = 20
lg_orders = [(p, l) for p in 0:(n_order ÷ 2) for l in (2 * p - n_order):(n_order - 2 * p)]
input_modes = stack(order -> LaguerreGaussian(w_in, order...)(xv, yv), lg_orders; dims = 3)
output_modes = stack(order -> LaguerreGaussian(w_in, order..., λ, z)(xv, yv), lg_orders;
                     dims = 3)

output_path = joinpath(data_dir, "../data/test_cases.npz")

npzwrite(output_path,
         Dict("input_modes" => input_modes,
              "output_modes" => output_modes,
              "wavelength" => λ,
              "z" => z,
              "dx" => ds[1],
              "dy" => ds[2]))
