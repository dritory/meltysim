using LinearAlgebra
using Random
using Plots


function predict(x, P, F, G, Q, u)
    # Extrapolate the state
    x_nxt = F*x + G*u
    # Extrapolate the covariance
    P_nxt = F*P*F' + Q
    return x_nxt, P_nxt
end

function update(x, P, z, h, h_jac, R)
    H = h_jac(x)
    # measurement Residual
    y = z - h(x)
    # Compute the innovation
    S = H*P*H' + R
    # Compute the Kalman gain
    K = P*H'*inv(S)
    # Identity matrix

    # Update estimate with measurement
    x_upd = x + K*y
    # Update the covariance
    P_upd = (I - K*H)*P*(I - K*H)' + K*R*K'
    
    return x_upd, P_upd
end


function kalman_filter(x0, P0, F, G, Q, h, h_jac, R, u, z)
    x = x0
    P = P0
    x_hist = [x0]
    P_hist = [P0]
    for z_i in z
        x, P = predict(x, P, F, G, Q, u)
        x, P = update(x, P, z_i, h, h_jac, R)
        push!(x_hist, x)
        push!(P_hist, P)
    end
    return x_hist, P_hist
end

function test_kalman_filter()
    f(t)= 1.0 * sin(0.1*t) + 2.
    
    dt = 0.1
    std_meas = 10.2
    std_acc = 0.25
    u = 2
    r = 0.1

    F = [1 dt; 0 1]
    G = [0.5*dt^2; dt]
    h = x -> [0; x[2]^2/r]
    h_jac = x -> [0 0 ; 0 2x[2]/r]
    Q = [dt^4/4 dt^3/2; dt^3/2 dt^2]*std_acc
    R = I(2)*std_meas^2
    x0 = [0.; 0.]
    P0 = [1. 0; 0. 1]

    z = [h([0; f(t)]) + [0; 100.0*rand()] for t in 0:dt:100]
    
    x_hist, P_hist = kalman_filter(x0, P0, F, G, Q, h, h_jac, R, u, z)
    return x_hist, P_hist, z, [f(t) for t in 0:dt:100]
end

x_hist, P_hist, z, x = test_kalman_filter()
plot([x[1] for x in x_hist], label="Estimate")
plot!([x[1] for x in x_hist], ribbon=[sqrt(P[1,1]) for P in P_hist], fillalpha=0.2, label="Uncertainty")
plot!([z_i[2] for z_i in z], label="Measurement")
plot!([x_i for x_i in x], label="True")

println("done")