using LinearAlgebra
using Distributions
using Random
using DrWatson

include(srcdir("filters.jl"))

function Q_func(observations, A, H, m0, P0, Q, R)
    kal = perform_kalman(observations, A, H, m0, P0, Q, R, lle = false)
    rts = perform_rts(kal, A, H, Q, R)
    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]

    Σ = zeros(size(P0))
    Φ = zeros(size(P0))
    B = zeros(size(observations[:, 1] * rts_means[:, 1]'))
    C = zeros(size(m0 * m0'))
    D = zeros(size(observations[:, 1] * observations[:, 1]'))

    K = size(observations, 2)

    for k = 2:K
        B += observations[:, k] * rts_means[:, k]'
        Σ += rts_covs[:, :, k] + (rts_means[:, k] * rts_means[:, k]')
        Φ += rts_covs[:, :, k-1] + (rts_means[:, k-1] * rts_means[:, k-1]')
        C += (rts_covs[:, :, k] * rts_G[:, :, k-1]') + (rts_means[:, k] * rts_means[:, k-1]')
        D += observations[:, k] * observations[:, k]'
    end
    B ./= K
    Σ ./= K
    Φ ./= K
    C ./= K
    D ./= K

    val_dict = @dict Σ Φ C B D
    return val_dict
end

function Q_func(observations, A, H, m0, P0, Q, R, _lp)
    kal = perform_kalman(observations, A, H, m0, P0, Q, R, lle = false)
    rts = perform_rts(kal, A, H, Q, R)
    rts_means = rts[1]
    rts_covs = rts[2]
    rts_G = rts[3]

    Σ = zeros(size(P0))
    Φ = zeros(size(P0))
    B = zeros(size(observations[:, 1] * rts_means[:, 1]'))
    C = zeros(size(m0 * m0'))
    D = zeros(size(observations[:, 1] * observations[:, 1]'))

    K = size(observations, 2)

    for k = 2:K
        B += observations[:, k] * rts_means[:, k]'
        Σ += rts_covs[:, :, k] + (rts_means[:, k] * rts_means[:, k]')
        Φ += rts_covs[:, :, k-1] + (rts_means[:, k-1] * rts_means[:, k-1]')
        C += (rts_covs[:, :, k] * rts_G[:, :, k-1]') + (rts_means[:, k] * rts_means[:, k-1]')
        D += observations[:, k] * observations[:, k]'
    end
    B ./= K
    Σ ./= K
    Φ ./= K
    C ./= K
    D ./= K

    _f1(A) = (K / 2.0) * tr(inv(Q) * (Σ - C * A' - A * C' + A * Φ * A'))
    _f2(A) = _lp(A)
    Qf(A) = _f1(A) + _f2(A)
    val_dict = @dict Σ Φ C B D
    return (Qf, _f1, _f2, val_dict)
end
