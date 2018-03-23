data = load('data.txt');
% Shuffle rows
iter = randperm(m);
data = data(iter, :);

X = data(:, 1); y = data(:, 2);
m = length(y);

% Add bias column 
X = [ones(m, 1), data(:,1)];



% Initialize theta
theta = rand(2, 1);

iterations = m;
alpha = 0.0001;

J = computeCost(X, y, theta);

fprintf('\nRunning Stochastic Gradient Descent ...\n')
% run gradient descent
[theta, theta_history, J_history] = SGD(@computeCost, X, y, theta, alpha);
fprintf('Theta found by stochastic gradient descent:\n');
fprintf('%f\n', theta);

figure;
plot(X(:,2), X*theta, '-');
hold on
scatter(X(:, 2), y);
hold off
pause
figure;
plot(1:iterations, J_history, '-');
pause
t1 = theta_history(:, 1);
t2 = theta_history(:, 2);

figure;
plot(t1, t2, '-');

