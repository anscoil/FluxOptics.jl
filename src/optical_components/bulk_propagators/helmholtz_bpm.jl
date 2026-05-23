struct HelmholtzBPM{M, C} <: AbstractPureComponent{M}
    trainability::Val{M}
    optical_components :: C
end

Functors.@functor HelmholtzBPM (optical_components,)

function HelmholtzBPM(u::HelmholtzField,
                      thickness::Real,
                      n_xyz::AbstractArray{<:Real, 3};
                      n0::Number = minimum(n_xyz),
                      method::Symbol = :Strang,
                      trainable::Bool = false,
                      buffered::Bool = false)
    n_slices = size(n_xyz, 3)
    @assert n_slices >= 2
    dz = thickness / n_slices
    M  = trainability(trainable, buffered)

    if method == :Strang  # Strang
        p_half = HelmholtzProp(u, dz/2; n0)
        p_full = HelmholtzProp(u, dz; n0)
        components = AbstractPipeComponent[p_half]
        for k in 1:n_slices
            push!(components, HelmholtzIndexSlice(u, dz, n0, n_xyz[:,:,k]; trainable, buffered))
            k < n_slices && push!(components, p_full)
        end
        push!(components, p_half)

    elseif method == :Yoshida  # Yoshida
        w1 = 1 / (2 - 2^(1/3))
        w0 = -2^(1/3) / (2 - 2^(1/3))
        p_start = HelmholtzProp(u, w1*dz/2; n0)
        p_full = HelmholtzProp(u, w1*dz; n0)
        p_mid = HelmholtzProp(u, (w1+w0)*dz/2; n0)
        components = AbstractPipeComponent[p_start]
        for k in 1:n_slices
            s_w1 = HelmholtzIndexSlice(u, w1*dz, n0, n_xyz[:,:,k]; trainable, buffered)
            s_w0 = HelmholtzIndexSlice(u, w0*dz, n0, n_xyz[:,:,k]; trainable, buffered)
            push!(components, s_w1, p_mid, s_w0, p_mid, s_w1)
            k < n_slices && push!(components, p_full)
        end
        push!(components, p_start)

    elseif method == :Suzuki  # Suzuki
        p  = 1 / (4 - 4^(1/3))
        p_start = HelmholtzProp(u, p*dz/2; n0)
        p_full = HelmholtzProp(u, p*dz; n0)
        p_mid = HelmholtzProp(u, (1-3p)*dz/2; n0)
        components = AbstractPipeComponent[p_start]
        for k in 1:n_slices
            s_p = HelmholtzIndexSlice(u, p*dz, n0, n_xyz[:,:,k]; trainable, buffered)
            s_mid = HelmholtzIndexSlice(u, (1-4p)*dz, n0, n_xyz[:,:,k]; trainable, buffered)
            push!(components, s_p, p_full, s_p, p_mid, s_mid, p_mid, s_p, p_full, s_p)
            k < n_slices && push!(components, p_full)
        end
        push!(components, p_start)

    else
        error("No such method $(method)")
    end

    HelmholtzBPM(Val(M), Tuple(components))
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

