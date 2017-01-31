function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(theta,1);
% calculate hypothesis do we need to add extra column bias term
h_theta = sigmoid(X*theta) ;

% calc expression 
main_vector = -y .* log(h_theta)  -  (1 - y) .* log(1 - h_theta) ; 
J =  (1/ m) *sum(main_vector);

% calculate square of theta 
r  = theta .^2 ;
% dont want to add theta_0 to regularized term
r(1) = 0;
% add regularization term to Cost function
J += (lambda/(2*m)) * sum(r);   


for featureIndex = 1: n
		%h += ((theta' * XT(:,featureIndex)) - y(featureIndex))*XT(:,featureIndex) ; 
		grad(featureIndex)  = (1/m) * sum( ( h_theta - y ) .*  X(:,featureIndex) );
		if( featureIndex != 1)
			grad(featureIndex)  +=  (lambda/m) * theta(featureIndex);
		endif;
		
end




% =============================================================

end
