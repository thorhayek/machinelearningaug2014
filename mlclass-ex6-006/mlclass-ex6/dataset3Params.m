function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%val_vector = [0.01, 0.03, 0.1, 0.3, 1,3,10,30];
%val_vector = [0.01, 0.03];
%len = length(val_vector);
%min_error = 999999999;
% for idx1 =  1:len

	% for idx2 = 1:len
	
		% %fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n C=%f \n sigman=%f \n',val_vector(idx1),val_vector(idx2));
		% model= svmTrain(X, y, val_vector(idx1), @(x1, x2) gaussianKernel(x1, x2, val_vector(idx2)));
		% predictions = svmPredict(model, Xval);
		% errorVal = mean(double(predictions ~= yval));
		% if(errorVal < min_error)
			% min_error = errorVal;
			% C_min = val_vector(idx1);
			% sigma_min = val_vector(idx2);
			% fprintf('\n minval = %f\n  C = %f \n sigma =%f \n',min_error, C_min, sigma_min);
		% endif
	
	% end;
% end; 
% fprintf('\n minval = %f\n  C = %f \n sigma =%f \n',min_error, C_min, sigma_min);
% C = C_min;
% sigma = sigma_min;

 C = 0.3;
 sigma = 0.1;

% =========================================================================

end
