function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %hypothesis func for each obs * val of each obs guve grad:
	H=X*theta; %size(X) = 97 x 2, size(theta) = 2 x 1, size(H) therefore is 97 x 1
	diffs = H-y; %vector sub= size(diffs) = 97 x 1
    %size(diffs') = 1 x 97
    %diffs' * X = 1 x 97 * 97 x 2 = 1 x 2 matrix (new thetas)
    thetasnew = diffs' * X;
    %(size(thetasnew) = 1x2)
    %size(theta) = 2 x 1
    %size(thetasnew') = 2x1
    %so substract (alpha/m)*thetasnew' from theta to give updated thetas:
    theta = theta - (alpha/m)*thetasnew';










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
