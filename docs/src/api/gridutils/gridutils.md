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

## Notes

- All spatial units should be consistent (typically micrometers)
- Rotations are counterclockwise for positive angles
- Transformations compose right-to-left: `(f ∘ g)(x) = f(g(x))`
- Offsets shift the zero position from grid center
