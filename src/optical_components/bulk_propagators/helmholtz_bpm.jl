struct HelmholtzBPM{M, C} <: AbstractPureComponent{M}
    trainability::Val{M}
    optical_components :: C
end

Functors.@functor HelmholtzBPM (optical_components,)

function HelmholtzBPM(u::HelmholtzField,
                      thickness::Real,
                      n0::Real,
                      n_xyz::AbstractArray{<:Number, 3};
                      trainable::Bool = false,
                      buffered::Bool = false)
    n_slices = size(n_xyz, 3)
    @assert n_slices >= 2
    dz = thickness / n_slices
    components = []
    p_half = HelmholtzProp(u, dz/2; n0)
    p = HelmholtzProp(u, dz; n0)
    push!(components, p_half)
    for k in 1:n_slices
        push!(components, HelmholtzIndexSlice(u, dz, n0, n_xyz[:,:,k];
                                              trainable, buffered))
        k < n_slices && push!(components, p)
    end
    push!(components, p_half)
    M = trainability(trainable, buffered)
    HelmholtzBPM(Val(M), components)
end

function propagate!(u::HelmholtzField, p::HelmholtzBPM)
    for pk in p.optical_components
        u = propagate!(u, pk)
    end
    u
end

function propagate(u::HelmholtzField, p::HelmholtzBPM)
    propagate!(copy(u), p)
end

