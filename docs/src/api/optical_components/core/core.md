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

## Technical Notes

### Trainability vs Buffering

**Trainability** determines if parameters can be optimized:
- `Static`: No optimization, no gradient tracking
- `Trainable`: Parameters can be optimized

**Buffering** determines memory allocation strategy for trainable components:
- `Unbuffered`: Allocates gradient buffers on-demand during backpropagation
- `Buffered`: Pre-allocates gradient and forward buffers at construction

**Performance trade-offs:**
```julia
# Unbuffered: lower memory, slower for many iterations
component = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false)

# Buffered: higher memory, faster for iterative optimization
component = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
```

### Direction Handling

Components implement bidirectional propagation:
- `Forward`: Default direction, positive z-axis
- `Backward`: Reverse direction, negative z-axis

Some components behave identically in both directions (e.g., phase masks), while others change behavior (e.g., amplifiers, gain media).

```julia
# Symmetric component (phase mask)
phase = Phase(u, (x, y) -> x^2)
u_fwd = propagate(u, phase, Forward)
u_bwd = propagate(u, phase, Backward)
# |u_fwd| == |u_bwd| (identical except for tilt handling)

# Asymmetric component (gain sheet)
gain = GainSheet(u, 2.0)  # 2x amplification
u_fwd = propagate(u, gain, Forward)   # Amplified
u_bwd = propagate(u, gain, Backward)  # Also amplified in backward
```

### Pure vs Custom Components

**Pure Components** (`AbstractPureComponent`):
- Functional interface: same input → same output
- No manual gradient rules needed
- Zygote handles differentiation automatically
- Can wrap complex internal state or `AbstractCustomComponent`
- Examples: `ASPropZ`, `OpticalSequence`, `FourierWrapper`

**Custom Components** (`AbstractCustomComponent`):
- Maintain explicit internal state
- Custom gradient implementations
- More control over memory and computation
- Better performance for specific operations
- Examples: `Phase`, `Mask`, `ASProp`, `BPM`

```julia
# Pure component: Zygote-compatible propagation
prop_pure = ASPropZ(u, 500.0; trainable=true)

# Custom component: explicit forward/backward passes
prop_custom = ASProp(u, 500.0; trainable=true)

# Both work in systems, but have different internal implementations
```

### Source vs Pipe Components

**Sources** (`AbstractOpticalSource`):
- Generate fields from nothing
- Take no input field
- Use `propagate(source)` (no input field argument)
- Placed at beginning of optical systems
- Example: `ScalarSource`

**Pipe Components** (`AbstractPipeComponent`):
- Transform existing fields
- Take input field as argument
- Use `propagate(u, component, direction)`
- Chained with `|>` operator
- Examples: `Phase`, `Mask`, `ASProp`, `FourierLens`

```julia
# Source: generates field
source = ScalarSource(u; trainable=true)
u_out = propagate(source)  # No input field needed

# Pipe: transforms field
phase = Phase(u, (x, y) -> x^2)
u_out = propagate(u, phase, Forward)  # Requires input field

# System composition
system = source |> phase |> propagator
result = system()  # source generates, components transform
```

### Memory and Performance

**Static components:**
- No gradient allocation
- Lower memory footprint
- Faster forward propagation
- Use for fixed optical elements

**Trainable, unbuffered:**
- Gradients allocated during backpropagation
- Lower base memory, higher peak memory
- Good for single-shot optimization or limited iterations
- Use when memory is tight

**Trainable, buffered:**
- Gradients and forward buffers pre-allocated
- Higher base memory, same peak memory
- Fastest for iterative optimization
- Use for production optimization loops

```julia
# Memory comparison for 128×128 field
u = ScalarField((128, 128), (2.0, 2.0), 1.064)

# Static: ~128 KB (field data only)
phase_static = Phase(u, (x, y) -> x^2)

# Trainable, unbuffered: ~128 KB base, ~256 KB peak
phase_unbuf = Phase(u, (x, y) -> 0.0; trainable=true, buffered=false)

# Trainable, buffered: ~384 KB (field + gradient + buffers)
phase_buf = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
```

### Component Data Access

```julia
# Get component data
phase = Phase(u, (x, y) -> 0.0; trainable=true)
φ = get_data(phase)  # Returns phase array

# Modify data directly (use with caution)
φ .= some_new_phase

# Get trainable parameters (for optimization)
params = trainable(phase)  # (; ϕ = φ)

# Collect from GPU
phase_gpu = Phase(u_gpu, (x, y) -> 0.0)
φ_cpu = collect(phase_gpu)  # Transfer to CPU
```

## Common Patterns

### Optimization Setup

```julia
using Zygote, Optimisers

# Create trainable components
u = ScalarField((128, 128), (2.0, 2.0), 1.064)
source = ScalarSource(u; trainable=true, buffered=true)
phase = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)
mask = Mask(u, (x, y) -> 1.0; trainable=true, buffered=true)
prop = ASProp(u, 500.0)

# Build system
system = source |> phase |> mask |> prop

# Setup optimization
target = ScalarField(target_data, (2.0, 2.0), 1.064)
metric = PowerCoupling(target)

loss() = 1.0 - metric(system())[]

# Extract trainable parameters
params = trainable(system)
opt_state = Optimisers.setup(Optimisers.Adam(0.01), params)

# Optimization loop
for iter in 1:1000
    l, grads = withgradient(loss, params)
    Optimisers.update!(opt_state, params, grads[1])
    
    if iter % 100 == 0
        println("Iteration $iter: loss = $l")
    end
end
```

### Bidirectional Propagation

```julia
# Forward and backward propagation through same system
u_input = ScalarField(input_data, (2.0, 2.0), 1.064)

# Forward pass
u_forward = propagate(u_input, component, Forward)

# Backward pass (e.g., for time-reversal or adjoint method)
u_backward = propagate(u_forward, component, Backward)

# For unitary components: u_backward ≈ u_input (up to numerical precision)
```

### Component Inspection

```julia
# Query component properties
comp = Phase(u, (x, y) -> 0.0; trainable=true, buffered=true)

istrainable(comp)  # true
isbuffered(comp)   # true
length(comp)       # 1 (atomic component)

# Extract information
data = get_data(comp)
params = trainable(comp)
data_cpu = collect(comp)

```

## See Also

- [Sources](../sources/index.md) for field generation components
- [Modulators](../modulators/index.md) for phase and amplitude modulation
- [Free-Space Propagators](../freespace/index.md) for propagation methods
- [System](../system/index.md) for building optical sequences and complex optical systems
- [OptimisersExt](../../optimisers/index.md) for optimization tools
