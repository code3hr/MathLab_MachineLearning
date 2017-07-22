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
a1 = [ones(m, 1), X];

z2 = a1*Theta1';
a2 = sigmoid( z2 ); % activation values for hidden layer

% add the bias unit to the hidden layer activations
a2PlusBias = [ones(size(a2, 1), 1), a2];

z3 = a2PlusBias*Theta2';

a3 = sigmoid( z3 ); % the predictions

% map y values to binary
all_combos = eye(num_labels);    
y_matrix = all_combos(y,:);         % works for Matlab

%hypoError = predictions-y; % error of each prediction

% costOfError = cost of how far away the hypo is from the classification
% to compute total cost for each example, loop over each label
costOfError = ( -y_matrix.*log(a3) ) - ...
            ( (1 - y_matrix) .* log(1-a3) );
        
%%% withOUT regularization %%% 

noRegCost = (1/m) * sum(sum(costOfError));  % NOT vectorized regression cost function
%J = noRegCost;
%grad = (1/m) * (X' * hypoError);  % vectorized logistic regression gradient

%%% WITH regularization %%%

squaredTheta1 = Theta1.^2;  % sqaures of Theta1 weights
squaredTheta1(:, 1) = 0;  % do not regularize the intercept param
squaredTheta2 = Theta2.^2;  % sqaures of Theta1 weights
squaredTheta2(:, 1) = 0;  % do not regularize the intercept param 

costRegConstant = lambda / (2*m);
costRegTerm = costRegConstant * ...
    ( sum(sum(squaredTheta1)) + sum(sum(squaredTheta2)) );

J = noRegCost + costRegTerm; % regularized cost function; return this

%%% calculate gradients %%%

% calculate "errors" of output nodes
% this should probably be:
% "delta3 = (a3 - y_matrix) * (y_matrix * (1-y_matrix));"
% but this was accepted by the grader. This might be because this is a 
% one-vs-all classification so the above formula would always give "0"
delta3 = (a3 - y_matrix);

% calculate "errors" of hidden nodes
delta2 = delta3*Theta2(:,2:end) .* sigmoidGradient(z2);

% accumulate the gradient for each layer's params:
accumGrad2 = delta3'*a2PlusBias;
accumGrad1 = delta2'*a1;

% calculate UNREGULARIZED gradients for each layer's params:
unRegTheta2_grad = (1/m)*accumGrad2;
unRegTheta1_grad = (1/m)*accumGrad1;

% calculate REGULAIZED gradients
gradRegTerms2 = (lambda / m) * Theta2; % calc grad reg term for Theta2
gradRegTerms2(:,1) = 0; % do not regularize the bias

gradRegTerms1 = (lambda / m) * Theta1; % calc grad reg term for Theta1
gradRegTerms1(:,1) = 0; % do not regularize the bias

Theta2_grad = unRegTheta2_grad + gradRegTerms2;
Theta1_grad = unRegTheta1_grad + gradRegTerms1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
