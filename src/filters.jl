using LinearAlgebra
using Distributions
using Random

function perform_kalman(observations, A, H, m0, P0, Q, R; lle = true)
    m = copy(m0)
    P = copy(P0)
    v = observations[:, 1] .- H * m
    S = H * P * H' .+ R
    K = P * H' / S
    T = size(observations, 2)
    _xd = length(m0)
    filtered_state = zeros(length(m0), T)
    filtered_cov = zeros(length(m0), length(m0), T)
    l_like_est = 0.0
    # offness = 0.0
    for t = 1:T
        m .= A * m
        P .= A * P * transpose(A) .+ Q
        v .= observations[:, t] .- H * m
        S .= H * P * transpose(H) .+ R
        # offness += norm(S - Matrix(Hermitian(S)), 1)
        # S .= Matrix(Hermitian(S))
        S .= 0.5 .* (S .+ S')
        K .= (P * transpose(H)) * inv(S)
        if lle
            l_like_est += logpdf(MvNormal(H * m, S), observations[:, t])
        end
        # unstable, need to implement in sqrt form
        m .= m + K * v
        P .= (I(_xd) .- K * H) * P * (I(_xd) .- K * H)' .+ K * R * K'
        filtered_state[:, t] .= m
        filtered_cov[:, :, t] .= P
    end
    return (filtered_state, filtered_cov, l_like_est)
end

function perform_rts(kalman_out, A, H, Q, R)
    kal_means = kalman_out[1]
    kal_covs = kalman_out[2]
    T = size(kal_means, 2)
    rts_means = zeros(size(kal_means))
    rts_covs = zeros(size(kal_covs))
    rts_means[:, T] = kal_means[:, T]
    rts_covs[:, :, T] = kal_covs[:, :, T]
    # just preallocation, values not important
    m_bar = A * kal_means[:, T]
    P_bar = A * kal_covs[:, :, T] * A' .+ Q
    G_ks = zeros(size(A)..., T)
    G_ks[:, :, T] .= kal_covs[:, :, T] * A' / P_bar
    for k = (T-1):-1:1
        @views m_bar .= A * kal_means[:, k]
        @views P_bar .= A * kal_covs[:, :, k] * A' .+ Q
        @views G_ks[:, :, k] .= kal_covs[:, :, k] * A' / P_bar
        @views rts_means[:, k] .= kal_means[:, k] .+ G_ks[:, :, k] * (rts_means[:, k+1] .- m_bar)
        @views rts_covs[:, :, k] .= kal_covs[:, :, k] .+ G_ks[:, :, k] * (rts_covs[:, :, k+1] .- P_bar) * G_ks[:, :, k]'
    end
    return (rts_means, rts_covs, G_ks)
end
