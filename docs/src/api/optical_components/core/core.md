# Core API

```@meta
CurrentModule = FluxOptics.OpticalComponents
```

## Abstract Types

### Component Hierarchy

```@docs
AbstractOpticalComponent
AbstractOpticalSource
AbstractPureSource
AbstractCustomSource
AbstractPipeComponent
AbstractPureComponent
AbstractCustomComponent
```

## Trainability System

### Types

```@docs
Trainability
Static
Trainable
Buffering
Buffered
Unbuffered
```

### Query Functions

```@docs
istrainable
isbuffered
```

## Propagation

### Direction Types

```@docs
Direction
Forward
Backward
```

### Propagation Functions

```@docs
propagate
propagate!
```

## Component Interface

```@docs
get_data
trainable(::AbstractOpticalComponent{Static})
```

## See Also

- [Sources](../sources/index.md) for field generation components
- [Modulators](../modulators/index.md) for phase and amplitude modulation
- [Free-Space Propagators](../freespace/index.md) for propagation methods
- [System](../system/index.md) for building optical sequences and complex optical systems
- [OptimisersExt](../../optimisers/index.md) for optimization tools
