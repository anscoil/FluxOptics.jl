const _probe_counter = Ref(0)

"""
    FieldProbe()

Create a probe to capture intermediate fields in an optical system.

Probes capture the field at specific points during propagation without affecting
the field itself. Useful for debugging, visualization, or monitoring optimization.

# Examples
```julia
# Create probes
probe1 = FieldProbe()
probe2 = FieldProbe()

# Insert in system
system = source |> component1 |> probe1 |> 
         component2 |> probe2 |> 
         component3

# Execute and access captured fields
result = system()
field1 = result.probes[probe1]
field2 = result.probes[probe2]
final = result.out
```

See also: [`OpticalSystem`](@ref)
"""
struct FieldProbe{M} <: AbstractPipeComponent{M}
    id::Int

    FieldProbe() = new{Static}(_probe_counter[] += 1)
end

reset_probe_counter!() = (_probe_counter[] = 0)

Functors.@functor FieldProbe ()

function propagate(u, p::FieldProbe, ::Type{<:Direction})
    (u, copy(u))
end
