# Active Media

Gain and amplification components.

## Overview

The `Active Media` module provides trainable **saturable gain sheets** for laser amplifiers.

## Quick Example

```julia
using FluxOptics

u = ScalarField(ones(ComplexF64, 256, 256), (2.0, 2.0), 1.064)

# Uniform gain sheet
gain = GainSheet(u, 0.1, 1e6, (x, y) -> 2.0)

# Pumped region (Gaussian)
gain_pumped = GainSheet(u, 0.1, 1e6, (x, y) -> 2.0 * exp(-(x^2 + y^2)/1000))

# Use in system
system = source |> phase |> gain |> propagator
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
