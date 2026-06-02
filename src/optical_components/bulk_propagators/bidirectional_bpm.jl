struct BidirectionalKernel{K, T}
    a::K
    exp_a_p::K
    exp_a_m::K
    nrm_f::T
end

Adapt.@adapt_structure BidirectionalKernel

struct BidirectionalBPM{M, N, K, E, T, P}  <: AbstractCustomComponent{M}
    trainability::Val{M}
    n_xyz::N
    n0::Complex{T}
    dz::T
    p_f::P
    E_plus::Union{Nothing, E}
    E_minus::E
    kernel::BidirectionalKernel{K, T}
    ∂p::Union{Nothing, @NamedTuple{E_minus::E}}
end

Functors.@functor BidirectionalBPM (n_xyz, E_minus)

trainable(p::BidirectionalBPM{<:Trainable}) = (; E_minus = p.E_minus)

get_preallocated_gradient(p::BidirectionalBPM{Trainable{Buffered}}) = p.∂p

function alloc_saved_buffer(u::HelmholtzField, p::BidirectionalBPM{Trainable{Unbuffered}})
    n_slices = size(p.n_xyz, 3)
    similar(u.electric, (size(u.electric)..., n_slices))
end

get_saved_buffer(p::BidirectionalBPM{Trainable{Buffered}}) = p.E_plus

function optimal_gauge(n_xyz, fill_factor=0.5)
    re_n2 = real.(n_xyz.^2)
    n0_real = sqrt(fill_factor * maximum(re_n2) + (1-fill_factor) * minimum(re_n2))
    n0_imag = 0.05 * n0_real
    return complex(n0_real, n0_imag)
end

function BidirectionalBPM(u::HelmholtzField{U},
                          thickness::Real,
                          n_xyz::AbstractArray{<:Number, 3};
                          n0::Number = optimal_gauge(n_xyz),
                          trainable::Bool = false,
                          buffered::Bool = false) where {T <: Real,
                                                         U <: AbstractArray{Complex{T}}}
    ns = size(u)[1:2]
    n_slices = size(n_xyz, 3)
    @assert size(n_xyz)[1:2] == ns
    @assert n_slices >= 2
    dz = T(thickness / n_slices)
    M  = trainability(trainable, buffered)
    N = isreal(n_xyz) ? T : Complex{T}
    n_xyz_buf = similar(u.electric, N, size(n_xyz))
    copyto!(n_xyz_buf, n_xyz)
    n0 = Complex{T}(n0)
    a = im * compute_kz(u, n0)
    exp_a_p = @. exp(a * dz)
    exp_a_m = @. exp(-a * dz)
    u_plan = similar(u.electric)
    p_f = make_fft_plans(u_plan, (1, 2); normalize = false)
    E_plus = (trainable && buffered) ? similar(u.electric,
                                               (size(u.electric)..., n_slices)) : nothing
    E_minus = trainable ? similar(u.electric, (size(u.electric)..., n_slices)) : nothing
    if !isnothing(E_plus)
        @. E_plus = 0
    end
    if !isnothing(E_minus)
        @. E_minus = 0
    end
    nrm_f = T(1/prod(ns))
    kernel = BidirectionalKernel(a, exp_a_p, exp_a_m, nrm_f)
    ∂p = (trainable && buffered) ? (; E_minus = similar(E_minus)) : nothing
    BidirectionalBPM(Val(M), n_xyz_buf, n0, dz, p_f, E_plus, E_minus, kernel, ∂p)
end

_get_val(A::AbstractArray{<:Any, 2}, I, J) = A[I]
function _get_val(A::AbstractArray{<:Any, N}, I, J) where {N}
    A[I, CartesianIndex(min.(Tuple(J), size(A)[3:end]))]
end

@kernel function propagate_helmholtz_forward_kernel!(electric, electric_dz, Ep, Em, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)
        E_val = electric[I, J]
        dE_val = electric_dz[I, J]
        E_minus = !isnothing(Em) ? Em[I, J] : zero(E_val)
        E_plus = 0.5 * (E_val + dE_val / a)
        E_plus *= exp_a_p
        E_minus *= exp_a_m
        electric[I,J] = nrm_f * (E_plus + E_minus)
        electric_dz[I,J] = nrm_f * a * (E_plus - E_minus)
        if !isnothing(Ep)
            Ep[I, J] = E_plus
        end
    end
end

@kernel function propagate_helmholtz_backward_kernel!(electric, electric_dz, ∂Em, kernel)
    nrm_f = kernel.nrm_f
    I = @index(Global, Cartesian)
    for J in CartesianIndices(axes(electric)[3:end])
        a = _get_val(kernel.a, I, J)
        exp_a_p = _get_val(kernel.exp_a_p, I, J)
        exp_a_m = _get_val(kernel.exp_a_m, I, J)

        ge = electric[I, J]
        gde = electric_dz[I, J]

        gE_plus_prop  = nrm_f * (ge + conj(a) * gde)
        gE_minus_prop = nrm_f * (ge - conj(a) * gde)

        if !isnothing(∂Em)
            ∂Em[I, J] = conj(exp_a_m) * gE_minus_prop
        end

        gE_plus_raw = conj(exp_a_p) * gE_plus_prop

        electric[I, J] = 0.5 * gE_plus_raw
        electric_dz[I, J] = 0.5 * conj(1 / a) * gE_plus_raw
    end
end

function propagate_slice_forward!(u::HelmholtzField, p::BidirectionalBPM,
                                  Ep::Union{Nothing, AbstractArray}, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    @. u.electric_dz += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric)
    compute_ft!(p.p_f, u)
    E_plus = !isnothing(Ep) ? selectdim(Ep, ndims(Ep), k) : nothing
    E_minus = !isnothing(p.E_minus) ? selectdim(p.E_minus, ndims(p.E_minus), k) : nothing
    propagate_helmholtz_forward_kernel!(backend)(
        u.electric, u.electric_dz, E_plus, E_minus, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
end

function propagate_slice_backward!(u::HelmholtzField, p::BidirectionalBPM,
                                   ∂Em::AbstractArray, k::Integer)
    backend = get_backend(u.electric)
    n_xy = view(p.n_xyz, :, :, k)
    compute_ft!(p.p_f, u)
    ∂E_minus = selectdim(∂Em, ndims(∂Em), k)
    propagate_helmholtz_backward_kernel!(backend)(
        u.electric, u.electric_dz, ∂E_minus, p.kernel;
        ndrange = size(u.electric)[1:2])
    compute_ift!(p.p_f, u)
    @. u.electric += ((2π/u.lambdas.val)^2 * (p.n0^2 - n_xy^2) * p.dz * u.electric_dz)
end

function _propagate!(u::HelmholtzField, p::BidirectionalBPM; Ep=nothing)
    slices_range = 1:size(p.n_xyz, 3)
    for k in slices_range
        propagate_slice_forward!(u, p, Ep, k)
    end
    u
end

propagate!(u::HelmholtzField, p::BidirectionalBPM) = _propagate!(u, p)

function propagate_and_save!(u::HelmholtzField,
                             Ep::AbstractArray,
                             p::BidirectionalBPM{<:Trainable})
    _propagate!(u, p; Ep)
end

function backpropagate_with_gradient!(∂v::HelmholtzField,
                                      Ep::AbstractArray,
                                      ∂p::NamedTuple,
                                      p::BidirectionalBPM{<:Trainable})
    slices_range = 1:size(p.n_xyz, 3)
    for k in reverse(slices_range)
        propagate_slice_backward!(∂v, p, ∂p.E_minus, k)
    end
    (∂v, ∂p)
end
