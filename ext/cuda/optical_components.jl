function OpticalComponents.make_nufft_plan(u::CuArray{Complex{T}, 2},
                                           ns::Tuple{Integer, Integer},
                                           s::Tuple{AbstractMatrix, AbstractMatrix},
                                           type::Integer,
                                           isign::Integer,
                                           eps::Real) where {T <: Real}
    p_nft = cufinufft_makeplan(type, [ns...], isign, 1, eps; dtype = T)
    cufinufft_setpts!(p_nft, s...)
    p_nft
end

function OpticalComponents.exec_nufft_plan!(p, u::CuArray{Complex{T}, 2}) where {T <: Real}
    nx, ny = size(u)
    u_in = reshape(u, (nx, ny, 1))
    u_out = reshape(u, (:, 1))
    if p.type == 2
        cufinufft_exec!(p, u_in, u_out)
    end
    if p.type == 1
        cufinufft_exec!(p, u_out, u_in)
    end
    if p.type == 3
        cufinufft_exec!(p, u_out, u_out)
    end
    u
end
