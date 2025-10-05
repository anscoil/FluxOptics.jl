# GridUtils API

## Spatial Vectors

```@docs
spatial_vectors
```

## 2D Transformations

```@docs
Shift2D
Rot2D
Id2D
```

## Examples

### Basic Coordinate Generation

```julia
# 1D coordinates
x, = spatial_vectors(64, 2.0)  # 64 points, 2.0 μm spacing

# 2D coordinates
xv, yv = spatial_vectors(128, 128, 1.5, 1.5)

# With offset (shift zero position)
xv, yv = spatial_vectors(128, 128, 1.5, 1.5; xc=10.0, yc=-5.0)
```

### Coordinate Transformations

```julia
# Simple translation
shift = Shift2D(20.0, -10.0)
new_point = shift([0.0, 0.0])  # [20.0, -10.0]

# Simple rotation (45°)
rot = Rot2D(π/4)
rotated = rot([1.0, 0.0])  # [√2/2, √2/2]

# Composition: translate then rotate
transform = Rot2D(π/6) ∘ Shift2D(5.0, 0.0)
result = transform([0.0, 0.0])
```

### Use with Modes

```julia
using FluxOptics

# Create offset and rotated Gaussian
xv, yv = spatial_vectors(128, 128, 2.0, 2.0)
gaussian = Gaussian(15.0)

# Apply transformation
transform = Shift2D(20.0, 10.0) ∘ Rot2D(π/8)
field = gaussian(xv, yv, transform)
```

## Notes

- All spatial units should be consistent (typically micrometers)
- Rotations are counterclockwise for positive angles
- Transformations compose right-to-left: `(f ∘ g)(x) = f(g(x))`
- Offsets shift the zero position from grid center
