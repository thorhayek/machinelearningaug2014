function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% calculate the value of hypothesis vector (m*1)( all training examples
% because we use vectorization )  
h = X*theta ;
squaredErrors  =  (h-y) .^2 ; % calculate squared error and then square each element of matric
% note A^2 calculated matrix square .^2 is element wise square
J = 1/(2*m) * sum(squaredErrors) ;



% =========================================================================

end;
