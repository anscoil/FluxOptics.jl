# Active Media

Gain and amplification components.

## Overview

The `Active Media` module provides trainable **saturable gain sheets** for laser amplifiers.

## Quick Example

```@example
using FluxOptics, CairoMakie

λ = 1.064
speckle_dist = generate_speckle((256, 256), (1.0, 1.0), λ, 0.05)
u = ScalarField(speckle_dist, (1.0, 1.0), λ)

# Pumped region (Gaussian)
gain_pumped = GainSheet(u, 1.5, 1.0, (x, y) -> exp(-((x-50)^2 + y^2)/50^2))

source = ScalarSource(u)

# Use in system
system = source |> gain_pumped

u_out = system().out

visualize((u, u_out), (intensity, complex); colormap=(:inferno, :dark), height=120)
```

## Key Types

- [`GainSheet`](@ref): Saturable gain medium

## See Also

- [Modulators](../modulators/index.md) for phase and amplitude modulation
- [Core](../core/index.md) for component interface

## Index

```@index
Modules = [FluxOptics.OpticalComponents]
Pages = ["active_media.md"]
Order = [:type, :function]
```
