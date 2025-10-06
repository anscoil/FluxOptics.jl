function pad!(u_pad::AbstractArray{T, N}, u::AbstractArray{T, N};
              offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
              pad_val = 0) where {T, N, Nd}
    @assert N >= Nd
    @assert all((>=)(0), offset)
    ns = size(u_pad)[1:Nd]
    for (k, n) in enumerate(offset)
        @assert size(u, k) + n <= ns[k]
    end
    pad_axes = ntuple(k -> k <= Nd ?
                           ((1 + offset[k]):(size(u, k) + offset[k])) :
                           axes(u, k), N)
    u_pad .= pad_val
    @views copyto!(u_pad[pad_axes...], u)
    u_pad
end

"""
    pad(array, new_size; offset=(0, 0, ...), pad_val=0)

Pad array to new size by placing original array at specified offset.

Creates a new array of `new_size` filled with `pad_val`, then copies the original
array starting at `offset`. The offset determines where the original array is positioned
in the padded result.

# Arguments
- `array`: Array to pad
- `new_size`: Target size (must be ≥ current size in each dimension)
- `offset`: Starting position for original array in padded result (default: all zeros)
- `pad_val`: Value for padded regions (default: 0)

# Returns
Padded array of size `new_size` with original array at specified offset.

# Examples
```julia
u = ones(ComplexF64, 128, 128)

# Place at origin (top-left)
u_pad = pad(u, (256, 256); offset=(0, 0))

# Center the array
offset_center = ((256-128)÷2, (256-128)÷2)
u_centered = pad(u, (256, 256); offset=offset_center)

# Offset to specific position
u_offset = pad(u, (256, 256); offset=(50, 30))
```

See also: [`crop`](@ref), [`PadCropOperator`](@ref)
"""
function pad(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
             offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
             pad_val = 0) where {T, N, Nd}
    shape = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
    u_pad = similar(u, shape)
    pad!(u_pad, u; offset, pad_val)
end

function crop!(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
               offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    @assert N >= Nd
    @assert all((>=)(0), offset)
    for (k, n) in enumerate(offset)
        @assert ns[k] + n <= size(u, k)
    end
    crop_axes = ntuple(k -> k <= Nd ?
                            ((1 + offset[k]):(ns[k] + offset[k])) :
                            axes(u, k), N)
    u_crop = view(u, crop_axes...)
end

function crop!(u_crop::AbstractArray{T, N}, u::AbstractArray{T, N};
               offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    ns = size(u_crop)[1:Nd]
    copyto!(u_crop, crop!(u, ns; offset))
    u_crop
end

"""
    crop(array, new_size; offset=(0, 0, ...))

Extract sub-region from array starting at specified offset.

Extracts a region of size `new_size` from the array, starting at `offset`.
The offset determines where extraction begins.

# Arguments
- `array`: Array to crop
- `new_size`: Size of region to extract (must be ≤ current size)
- `offset`: Starting indices for extraction (default: all zeros)

# Returns
Cropped array of size `new_size`.

# Examples
```julia
u_large = ones(ComplexF64, 256, 256)

# Extract from origin
u_crop = crop(u_large, (128, 128); offset=(0, 0))

# Extract centered region
offset_center = ((256-128)÷2, (256-128)÷2)
u_center = crop(u_large, (128, 128); offset=offset_center)

# Extract from specific position
u_region = crop(u_large, (128, 128); offset=(50, 30))
```

See also: [`pad`](@ref), [`PadCropOperator`](@ref)
"""
function crop(u::AbstractArray{T, N}, ns::NTuple{Nd, Integer};
              offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {T, N, Nd}
    shape = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
    u_crop = similar(u, shape)
    copyto!(u_crop, crop!(u, ns; offset))
    u_crop
end

struct PadOperator{U, Nd, T, N}
    u_tmp::Union{Nothing, U}
    size_in::NTuple{N, Int}
    size_out::NTuple{N, Int}
    size_pad::NTuple{Nd, Int}
    offset::NTuple{Nd, Int}
    pad_value::T

    function PadOperator(u::U, ns::NTuple{Nd, Integer};
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {T, N, U <: AbstractArray{T, N}, Nd}
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
        new{U, Nd, T, N}(nothing, size_in, size_out, ns, offset, T(pad_val))
    end

    function PadOperator(u::U, u_tmp::U, nd::Integer;
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, nd),
                         pad_val = 0) where {T, N, U <: AbstractArray{T, N}, Nd}
        @assert Nd == nd
        ns = size(u_tmp)[1:Nd]
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), N)
        new{U, Nd, T, N}(u_tmp, size_in, size_out, ns, offset, T(pad_val))
    end

    function PadOperator(u::ScalarField{U, Nd}, ns::NTuple{Nd, Integer};
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {U, Nd}
        PadOperator(u.electric, ns; offset, pad_val)
    end

    function PadOperator(u::ScalarField{U, Nd}, u_tmp::U;
                         offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                         pad_val = 0) where {U, Nd}
        PadOperator(u.electric, u_tmp, Nd; offset, pad_val)
    end
end

function (p::PadOperator{U, Nd})(u::ScalarField{U, Nd}) where {U, Nd}
    electric_pad = if isnothing(p.u_tmp)
        pad(u.electric, p.size_pad; offset = p.offset, pad_val = p.pad_value)
    else
        pad!(p.u_tmp, u.electric; offset = p.offset, pad_val = p.pad_value)
    end
    set_field_data(u, electric_pad)
end

struct CropOperator{U, Nd, N}
    u_tmp::Union{Nothing, U}
    size_in::NTuple{N, Int}
    size_out::NTuple{N, Int}
    size_crop::NTuple{Nd, Int}
    offset::NTuple{Nd, Int}

    function CropOperator(u::U, ns::NTuple{Nd, Integer};
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), ndims(u))
        new{U, Nd, ndims(u)}(nothing, size_in, size_out, ns, offset)
    end

    function CropOperator(u::U, u_tmp::U, nd::Integer;
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, nd)) where {U, Nd}
        @assert Nd == nd
        ns = size(u_tmp)[1:Nd]
        size_in = size(u)
        size_out = ntuple(k -> k <= Nd ? ns[k] : size(u, k), ndims(u))
        new{U, Nd, ndims(u)}(u_tmp, size_in, size_out, ns, offset)
    end

    function CropOperator(u::ScalarField{U, Nd}, ns::NTuple{Nd, Integer};
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        CropOperator(u.electric, ns; offset)
    end

    function CropOperator(u::ScalarField{U, Nd}, u_tmp::U;
                          offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd)) where {U, Nd}
        CropOperator(u.electric, u_tmp, Nd; offset)
    end
end

function (p::CropOperator{U, Nd})(u::ScalarField{U, Nd}) where {U, Nd}
    electric_crop = if isnothing(p.u_tmp)
        crop(u.electric, p.size_crop; offset = p.offset)
    else
        crop!(p.u_tmp, u.electric; offset = p.offset)
    end
    set_field_data(u, electric_crop)
end

function (p::CropOperator{U, Nd})(u::ScalarField{U, Nd}, u_tmp::U) where {U, Nd}
    crop!(u_tmp, u.electric; offset = p.offset)
    set_field_data(u, u_tmp)
end

"""
    PadCropOperator(u::ScalarField, u_tmp::AbstractArray; offset=(0,...), pad_val=0, store_ref=false)

Create reversible pad/crop operator for efficient memory management.

Wraps padding and cropping as an optical component. The `adjoint` operation swaps
pad and crop directions. With `store_ref=true`, can reuse the original array during
crop (avoids allocation) when pad/crop are symmetric.

# Arguments
- `u::ScalarField`: Original (small) field template
- `u_tmp`: Padded array template (defines target size)
- `offset`: Position offset for pad/crop (default: all zeros)
- `pad_val`: Value for padded regions (default: 0)
- `store_ref::Bool`: Store reference to enable zero-allocation crop (default: false)

# Usage Patterns

**Basic usage (with allocation):**
```julia
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
u_tmp = zeros(ComplexF64, 256, 256)

pad_op = PadCropOperator(u, u_tmp)
crop_op = adjoint(pad_op)

# Pad then crop
system = source |> pad_op |> propagator |> crop_op
```

**Zero-allocation mode (symmetric pad/crop):**
```julia
# Store reference to reuse original array during crop
pad_op = PadCropOperator(u, u_tmp; store_ref=true)
crop_op = adjoint(pad_op)

# Crop reuses original array - no allocation
system = source |> pad_op |> processing |> crop_op
```

**With offset:**
```julia
# Center the field in padded array
offset = ((256-128)÷2, (256-128)÷2)
pad_op = PadCropOperator(u, u_tmp; offset=offset)
```

# Technical Details

- `Forward` direction: Applies pad if `ispad=true`, crop otherwise
- `Backward` direction: Swaps operation (crop if `ispad=true`, pad otherwise)
- `adjoint(op)`: Flips `ispad` flag, swapping pad/crop behavior
- `store_ref=true`: Only works for symmetric operations (same offset, sizes match)

# Examples

```julia
# Avoid aliasing in propagation
u = ScalarField(ones(ComplexF64, 128, 128), (2.0, 2.0), 1.064)
u_tmp = zeros(ComplexF64, 256, 256)

pad_op = PadCropOperator(u, u_tmp; store_ref=true)
prop = ASProp(set_field_data(u, u_tmp), 1000.0)
crop_op = adjoint(pad_op)

# Efficient: crop reuses original buffer
system = ScalarSource(u) |> pad_op |> prop |> crop_op

result = system().out.electric
```

See also: [`pad`](@ref), [`crop`](@ref)
"""
struct PadCropOperator{M, U, Nd, T, N} <: AbstractCustomComponent{M}
    p_pad::PadOperator{U, Nd, T, N}
    p_crop::CropOperator{U, Nd, N}
    u_tmp::Ref{U}
    ispad::Bool
    store_ref::Bool

    function PadCropOperator(p_pad::PadOperator{U, Nd, T, N},
                             p_crop::CropOperator{U, Nd, N},
                             u_tmp::Ref{U},
                             ispad::Bool; store_ref::Bool = false) where {U, Nd, T, N}
        new{Static, U, Nd, T, N}(p_pad, p_crop, u_tmp, ispad, store_ref)
    end

    function PadCropOperator(u::ScalarField{U, Nd}, u_tmp::U;
                             offset::NTuple{Nd, Integer} = ntuple(_ -> 0, Nd),
                             pad_val = 0,
                             store_ref::Bool = false) where {T, N, Nd,
                                                             U <: AbstractArray{T, N}}
        p_pad = PadOperator(u, u_tmp; offset, pad_val)
        p_crop = CropOperator(u_tmp, size(u)[1:Nd])
        new{Static, U, Nd, T, N}(p_pad, p_crop, Ref{U}(), true, store_ref)
    end
end

function Base.adjoint(p::PadCropOperator{M, U}) where {M, U}
    PadCropOperator(p.p_pad, p.p_crop, p.u_tmp, !p.ispad; store_ref = p.store_ref)
end

Functors.@functor PadCropOperator ()

function propagate!(u::ScalarField{U, Nd}, p::PadCropOperator{M, U, Nd},
                    ::Type{Forward}) where {M, U, Nd}
    if p.ispad
        if p.store_ref
            p.u_tmp[] = u.electric
        end
        p.p_pad(u)
    else
        if p.store_ref
            p.p_crop(u, p.u_tmp[])
        else
            p.p_crop(u)
        end
    end
end

function propagate!(u::ScalarField{U, Nd}, p::PadCropOperator{M, U, Nd},
                    ::Type{Backward}) where {M, U, Nd}
    if !p.ispad
        if p.store_ref
            p.u_tmp[] = u.electric
        end
        p.p_pad(u)
    else
        if p.store_ref
            p.p_crop(u, p.u_tmp[])
        else
            p.p_crop(u)
        end
    end
end

function backpropagate!(∂v::ScalarField, p::PadCropOperator, direction::Type{<:Direction})
    propagate!(∂v, p, reverse(direction))
end
