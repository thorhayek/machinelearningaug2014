function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% calculate the value of hypothesis vector (m*1)( all training examples
% because we use vectorization )  
h = X*theta ; % size m*1
errors = h-y ;  % m*1
squaredErrors  =  errors .^2 ; % calculate squared error and then square each element of matric
% note A^2 calculated matrix square .^2 is element wise square
J = 1/(2*m) * sum(squaredErrors) ;

thetaSquared =  theta .^2 ; 
thetaSquared(1) = 0 ;
J +=  (lambda/(2*m))  * sum(thetaSquared);

%fprintf( " Theta %f... grad %f \n",size(theta),size(grad));
%theta
%grad
% calculate the gradient 
	for thetai = 1:size(theta)
		for instance = 1:m
			grad(thetai) =  grad(thetai) +( errors(instance) * X(instance,thetai) ); 
		end
		
		grad(thetai) = (1/m)  * grad(thetai);	
		% add regularization term for j>1
		if( thetai > 1)
			grad(thetai) =  grad(thetai) + (lambda/m) * theta(thetai) ;
		endif;
	end
	
	%fprintf( " The  value of cost function %f... in iter %f \n",computeCost(X,y,theta),iter );
    % h += ((theta' * XT(:,featureIndex)) - y(featureIndex))*XT(:,featureIndex) ;



% =========================================================================

grad = grad(:);

end
