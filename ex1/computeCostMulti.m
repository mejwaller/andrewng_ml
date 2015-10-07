function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%H(theta) = theta0*theta1*x
%size(X) = 97 x 2 (97 obs, '1' for x0, and data(:,1) for x1
%size(theta) = 2*1
%so vectorizing, multiply X and theta giving a 97 x 1 vector of H for each obs:

H = X * theta;
 %y (observed data) is 97 x 1 so vector sub H-y gves diffs of predicted -observed.
diffs = H-y;
%square the diffs (square error f)
sqrd=diffs.^2;

%sum the results for each obs (m of them) and multiply result by 1/(2m) for cost function J:

J= (1/(2*m))*sum(sqrd);




% =========================================================================

end
