function [theta, theta_history, J_history] = SGD(J, X, y, theta, alpha)
    m = length(y);
    J_history = zeros(m, 1);
    n = length(theta);
    theta_history = zeros(m, n);
    delta = zeros(n, 1);
    epsilon = 0.001;
    
    for iter = 1:m
        for j = 1:n
            temp1 = theta;
            temp2 = theta;
            temp1(j) = temp1(j) + epsilon;
            temp2(j) = temp2(j) - epsilon;
            delta(j) = (J(X(iter, :), y(iter), temp1) - J(X(iter, :), y(iter), temp2))/(2 * epsilon);
        end
        
        theta = theta - alpha * delta;
        theta_history(iter, :) = theta';
        J_history(iter) = J(X, y, theta);
    end
end