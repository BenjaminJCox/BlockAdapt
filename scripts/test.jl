using Distributions
using Plots

include(srcdir("filters.jl"))

Random.seed!(0x8d8e1b4c2169a71cfde)

A = [0.8 0.2 0.0; 0.0 0.7 0.3; 0.1 0.0 0.9]
a_dim = 3
Q = Matrix(1.0^2 .* I(a_dim))
R = Matrix(1.0^2 .* I(a_dim))
H = Matrix(1.0 .* I(a_dim))
P = Matrix(1e-8 .* I(a_dim))

m0 = ones(a_dim)

T = 100

X = zeros(a_dim, T)
Y = zeros(a_dim, T)

prs_noise = MvNormal(Q)
obs_noise = MvNormal(R)
prior_state = MvNormal(m0, P)

X[:, 1] = rand(prior_state)
Y[:, 1] = H * X[:, 1] .+ rand(obs_noise)

for t = 2:T
    X[:, t] = A * X[:, t-1] .+ rand(prs_noise)
    Y[:, t] = H * X[:, t] .+ rand(obs_noise)
end

true_filtered = perform_kalman(Y, A, H, m0, P, Q, R)
rts_sm = perform_rts(true_filtered, A, H, Q, R)

vois = 1:a_dim
plot_arr = Array{Any,1}(undef, length(vois))
for voi in vois
    plot_series_true = true_filtered[1][voi, :]
    plot_series_opt = rts_sm[1][voi, :]
    plot_obs = X[voi, :]

    p1 = plot(plot_obs, label = "Truth")
    plot!(plot_series_true, label = "True Filter")
    plot!(plot_series_opt, label = "True Smoother")
    plot_arr[voi] = p1
end

plot(plot_arr..., size = (1000, 750), layout = (:, 1), legend = :outerright)
