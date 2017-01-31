function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix; Adds a column to the left of x
X = [ones(m, 1) X];

% create a new vector that will help us recode the y values
y_recoded = zeros(num_labels,1);

%fprintf( "%f \n ",size(y)); % y = 5000 * 1 

sum_d1 = zeros(size(Theta1));
sum_d2 = zeros(size(Theta2));

% calculate The Forward Propagation / FeedForward network 
% output is hthetax 10*1 ( numlabels * 1) 
for instance=1:m
		
		% calculate the feed forward propagation for this instance 
		a1 = X(instance,:)'; % size of vector  401*1 
		z2 = Theta1*a1;  % z2 size = 25 *1 
		a2 = sigmoid(z2); % size = 25 * 1 
		% add 1 as bias term to a2 
		a2 = [1; a2];   % 26*1
		z3 = Theta2*a2; % z3 = 10*1
		a3 = sigmoid(z3); % 10*1
		h_theta_x = a3  ; % 10*1 vector
		
		
		
		% calculate the cost function for this instance 
		y_recoded(y(instance)) = 1;  % creates a 10*1 vector with all zeros and ith row set to 1;
		main_vector = -y_recoded .* log(h_theta_x)  - (1-y_recoded) .* log(1- h_theta_x);
		J += sum(main_vector);
		
		
		% Calculate smalldelta for gradient calculation 
		d3 = a3 - y_recoded ; % 10*1
		%Theta2_squared = 
		d2 = ((Theta2)' * d3 )(2:end) .* sigmoidGradient(z2); % 25*1
		
		% include the errors for bias term ; they dont matter 
		%a2 = a2(2:end);
		%a1 = a1(2:end);
		sum_d2 = sum_d2 + d3*(a2)';
		sum_d1 = sum_d1 + d2*(a1)'; 
		
		Theta1_grad = 1/m .* sum_d1;
		Theta2_grad = 1/m .* sum_d2;
		
		y_recoded(y(instance)) = 0;  %creates a 10*1 vector with all zeros and ith row set to 0;
end;

J = J ./ m ; 

% Set the first column of each matrix to 0 
% because 1st column bias terms are not regularized 
Theta1(:,1) = 0;
Theta2(:,1) = 0;

regTheta1 = 0 ;
regTheta2 = 0 ; 
 
% regularizing Theta1 
%i = size(Theta1,1);
%j = size(Theta1,2);

regTheta1 = (Theta1 .^2);
% regularizing Theta2 
%i = size(Theta2,1);
%j = size(Theta2,2);

regTheta2 = (Theta2 .^2);

% add Regularization ters 
% divide each element by m to get Cost Function 
J +=  (lambda/(2*m))*( sum(sum(regTheta1)) + sum(sum(regTheta2))) ; 

% calculate the Cost Function 
%for instance=1:m
%end;




%n = size(theta,1);
% calculate hypothesis do we need to add extra column bias term
%h_theta = sigmoid(X*theta) ;

% calc expression 
%main_vector = -y .* log(h_theta)  -  (1 - y) .* log(1 - h_theta) ; 

%J =  (1/ m) *sum(main_vector);

% calculate square of theta 

%r  = theta .^2 ;
% dont want to add theta_0 to regularized term
%r(1) = 0;
% add regularization term to Cost function
%J += (lambda/(2*m)) * sum(r);   


		















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
