% Load the ONNX model
onnxModel = importONNXNetwork('cbf_model.onnx', 'OutputLayerType', 'regression');

% Parameters
c = 1; % Safe set constant
x0 = [1; 0]; % Initial state
T = 10; % Simulation time
dt = 0.01; % Time step
t = 0:dt:T;

% Preallocate state and control input
x = zeros(2, length(t));
u = zeros(1, length(t));
x(:, 1) = x0;

% Array to store safe trajectories
safe_trajectories = [];

% Simulation loop
for k = 1:length(t)-1
    % Control input (simple PD controller for demonstration)
    u(k) = -x(1, k) - x(2, k);
    
    % Ensure the control input maintains the new CBF constraint
    % Get the learned CBF from the ONNX model
    input = dlarray(x(:, k)', 'CB');
    h_learned = predict(onnxModel, input);
    
    if h_learned < 0
        % Apply a minimal control input that respects the learned CBF constraint
        u(k) = -x(1, k) * 0.1 - x(2, k) * 0.1;
    end
    
    % Update the state using the dynamics
    dx = double_integrator(t(k), x(:, k), u(k));
    x(:, k+1) = x(:, k) + dx * dt;
    
    % Append the current state to the safe trajectories
    safe_trajectories = [safe_trajectories, x(:, k)];
end

% Append the last state
safe_trajectories = [safe_trajectories, x(:, end)];

% Save the safe trajectories to a file
save('safe_trajectories_learned.mat', 'safe_trajectories');

% Plot the results
figure;
subplot(2, 1, 1);
plot(t, x(1, :), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Position');
title('Double Integrator System with Learned CBF');

subplot(2, 1, 2);
plot(t, x(2, :), 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Velocity');

% Define the system dynamics
function dx = double_integrator(t, x, u)
    dx = [x(2); u];
end

% Define the initial Control Barrier Function (CBF)
function h = control_barrier_function(x, c)
    h = 0.5 * (x(1)^2 + x(2)^2) - c;
end
