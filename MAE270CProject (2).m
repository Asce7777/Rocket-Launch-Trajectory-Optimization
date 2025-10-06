%% MAE 270C: Rocket Optimization
clear; clc; close all;

%% Parameters 
g = 32;
T = 2 * g;
tf = 900;
x_init = [0; 0; 0; 0];
target_h = 320000;
target_w = 0;

%% Numerical Function Solver (Part A)
fprintf('Num Fun Solv Method\n');

opts = optimoptions('fsolve','Display','iter','OutputFcn', @store_nu_history);

u_fn = @(t,nu) atan( -nu(2) - (tf - t) * nu(1) );

nu_guess = [0; 1];

nu_solution = fsolve(@(nu) residual_fn(nu, tf, x_init, g, T, target_h, target_w, u_fn), nu_guess, opts);

% Simulate optimal trajectory
u_star = @(t) atan( -nu_solution(2) - (tf - t) * nu_solution(1) );
[t_star, x_star] = ode45(@(t,x) rocket_ode(x, u_star(t), g, T), [0 tf], x_init);

% Hamiltonian computation
H_vals = zeros(length(t_star),1);
for i = 1:length(t_star)
    lam1 = 0;
    lam2 = nu_solution(1);
    lam3 = -1;
    lam4 = nu_solution(2) + (tf - t_star(i)) * nu_solution(1);
    
    u_now = u_star(t_star(i));
    dx3 = T * cos(u_now);
    dx4 = T * sin(u_now) - g;
    
    H_vals(i) = lam1 * x_star(i,3) + lam2 * x_star(i,4) + lam3 * dx3 + lam4 * dx4;
end

%% Plot (Part A)
figure;
plot(t_star, H_vals);
xlabel('Time [s]'); ylabel('Hamiltonian');
title('Hamiltonian along optimal trajectory');
grid on;

% Load nu history from OutputFcn
history_nu = evalin('base','history_nu');

figure;
plot(0:length(history_nu)-1, history_nu(:,1), 'o-'); 
hold on;
plot(0:length(history_nu)-1, history_nu(:,2), 's-');
xlabel('Iteration'); ylabel('Nu values');
legend('\nu_2','\nu_4');
title('Num Fun Solv Convergence');
grid on;

%% Steepest Descent (Part B)
fprintf('\nSteepest Descent Method\n');

A_mat = [0 0 1 0;
         0 0 0 1;
         0 0 0 0;
         0 0 0 0];

psi_x_tf = [0 1 0 0; 0 0 0 1];
phi_x_tf = [0 0 -1 0];

dt_B = 1;
time_grid = 0:dt_B:tf;
num_grid = length(time_grid);

u_guess = atan( -nu_guess(2) - (tf - time_grid) * nu_guess(1) );

lambda_psi = zeros(4,2,num_grid);
lambda_psi(:,:,num_grid) = psi_x_tf';
for j = num_grid-1:-1:1
    lambda_psi(:,:,j) = lambda_psi(:,:,j+1) - dt_B * ( -A_mat' * lambda_psi(:,:,j+1) );
end

lambda_phi = zeros(4,num_grid);
lambda_phi(:,num_grid) = phi_x_tf';
for j = num_grid-1:-1:1
    lambda_phi(:,j) = lambda_phi(:,j+1) - dt_B * ( -A_mat' * lambda_phi(:,j+1) );
end

max_iter = 300;
tol = 1e-6;
eps_step = 5e-3;

psi_hist = zeros(max_iter, 2);
phi_hist = zeros(max_iter, 1);

u_traj = u_guess;

for iter = 1:max_iter
    [~, x_traj] = ode45(@(t,x) rocket_ode(x, interp1(time_grid, u_traj, t), g, T), [0 tf], x_init);
    x_traj = x_traj';
    x_final = x_traj(:, end);
    
    psi_curr = [x_final(2) - target_h;
                x_final(4) - target_w];
    phi_curr = -x_final(3);
    
    psi_hist(iter, :) = psi_curr';
    phi_hist(iter) = phi_curr;
    
    if all(abs(psi_curr) < tol) && all(abs(phi_hist(iter)-phi_hist(iter-1)) < tol)
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    delta_psi = -0.15 * psi_curr;
    
    f_u_grid = zeros(4, num_grid);
    f_u_grid(3,:) = -T * sin(u_traj);
    f_u_grid(4,:) = T * cos(u_traj);
    
    integ1 = zeros(2,2);
    integ3 = zeros(2,1);
    
    for j = 1:num_grid
        L_psi = lambda_psi(:,:,j);
        L_phi = lambda_phi(:,j);
        fu_j  = f_u_grid(:,j);
        
        integ1 = integ1 + (L_psi' * fu_j) * (fu_j' * L_psi) * dt_B;
        integ3 = integ3 + (L_psi' * fu_j) * (fu_j' * L_phi) * dt_B;
    end
    
    nu_update = -integ1 \ (delta_psi / eps_step + integ3);
    
    du = zeros(1,num_grid);
    for j = 1:num_grid
        L_psi = lambda_psi(:,:,j);
        L_phi = lambda_phi(:,j);
        fu_j  = f_u_grid(:,j);
        
        du(j) = -eps_step * (L_phi' + nu_update' * L_psi') * fu_j;
    end
    
    u_traj = u_traj + du;
end

fprintf('Steepest Descent completed in %d iterations\n', iter);

%% Plotting Steepest Descent (Part B)
k_axis = 1:iter;

% Plot Cost History
figure;
plot(k_axis, phi_hist(1:iter));
xlabel('Iteration'); ylabel('Cost \phi');
title('Cost history - Steepest Descent');
grid on;

% Plot Psi(h(tf)) over Iterations
figure;
plot(k_axis, psi_hist(1:iter,1));
xlabel('Iteration'); ylabel('\psi(h(t_f))');
title('\psi(h(t_f)) Values vs Iterations');
ylim([-1e-5, 1e-5]);
xlim([0,iter(end)]);
grid on;

% Plot Psi(w(tf)) over Iterations
figure;
plot(k_axis, psi_hist(1:iter,2));
xlabel('Iteration'); ylabel('\psi(w(t_f))');
title('\psi(w(t_f)) Values vs Iterations');
ylim([-1e-5, 1e-5]);
xlim([0,iter(end)]);
grid on;

% Input comparison
figure;
plot(t_star, u_star(t_star)); hold on;
plot(time_grid, u_traj);
legend('Num Fun Solv','Steepest Descent');
xlabel('Time [s]'); ylabel('u(t)');
grid on;

%% State and Input Comparison Plots

% Trajectory obtained for Par
[tB_traj, xB_traj] = ode45(@(t,x) rocket_ode(x, interp1(time_grid, u_traj, t), g, T), [0 tf], x_init);

% Plot r(t)
figure;
plot(t_star, x_star(:,1)); hold on;
plot(tB_traj, xB_traj(:,1));
legend('Num Fun Solv', 'Steepest Descent');
xlabel('Time [s]'); ylabel('Horizontal Position r(t) [ft]');
grid on;

% Plot h(t)
figure;
plot(t_star, x_star(:,2)); hold on;
plot(tB_traj, xB_traj(:,2));
yline(target_h, 'k--');
legend('Num Fun Solv', 'Steepest Descent', 'Target h_f','Location', 'southeast');
xlabel('Time [s]'); ylabel('Vertical Position h(t) [ft]');
grid on;

% Plot v(t)
figure;
plot(t_star, x_star(:,3)); hold on;
plot(tB_traj, xB_traj(:,3));
legend('Num Fun Solv', 'Steepest Descent','Location', 'southeast');
xlabel('Time [s]'); ylabel('Horizontal Velocity v(t) [ft/s]');
grid on;

% Plot w(t)
figure;
plot(t_star, x_star(:,4)); hold on;
plot(tB_traj, xB_traj(:,4));
yline(target_w, 'k--');
legend('Num Fun Solv', 'Steepest Descent', 'Target w_f');
xlabel('Time [s]'); ylabel('Vertical Velocity w(t) [ft/s]');
grid on;

% Plot h(t) vs r(t)
figure;
plot(x_star(:,1), x_star(:,2)); 
hold on;
plot(xB_traj(:,1), xB_traj(:,2));
yline(target_h, 'k--');
legend('Num Fun Solv','Steepest Descent','Target h_f','Location', 'southeast');
xlabel('r [ft]'); ylabel('h [ft]');
grid on;

%% Numeric Calculations 
% Part A
dH_A = x_star(end,2) - target_h;   % delta_h
dW_A = x_star(end,4) - target_w;   % delta_w

% Part B  (values from last SD iteration)
dH_B = psi_curr(1);
dW_B = psi_curr(2);

fprintf('\nPART A  ν2 = %.15g   ν4 = %.15g   delta_h = %.15g   delta_w = %.15g\n', ...
        nu_solution(1), nu_solution(2), dH_A, dW_A);

fprintf('PART B  ν2 = %.15g   ν4 = %.15g   delta_h = %.15g   delta_w = %.15g   iters = %d\n\n', ...
        nu_update(1), nu_update(2), dH_B, dW_B, iter);

%% Helper functions 
% Rocket ODE
function dx = rocket_ode(x, u, g, T)
    dx = [ x(3);
           x(4);
           T * cos(u);
           T * sin(u) - g ];
end

% Residual
function err = residual_fn(nu, tf, x_init, g, T, target_h, target_w, u_fn)

    [~, x_sim] = ode45(@(t,x) rocket_ode(x, u_fn(t,nu), g, T), [0 tf], x_init);
    
    x_tf = x_sim(end,:);
    
    err = [ x_tf(2) - target_h;
            x_tf(4) - target_w ];
end

% Store nu history
function stop = store_nu_history(x,~,state)
    persistent hist_nu
    if strcmp(state,'init')
        hist_nu = x(:)';
    elseif strcmp(state,'iter')
        hist_nu(end+1,:) = x(:)';
    elseif strcmp(state,'done')
        assignin('base','history_nu', hist_nu);
    end
    stop = false;
end
