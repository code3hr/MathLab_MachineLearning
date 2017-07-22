function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% x1 = mean(X_norm(:,1));
% 
% x2 = mean(X_norm(:,2));
% 
% mu  = [X_norm(:,1)-x1, X_norm(:,2)-x2];
% 
% 
% 
% std1 = std(X_norm(:,1));
% std2 = std(X_norm(:,2));
% 
% sigma =[X_norm(:,1)./std1, X_norm(:,2)./std2];

% works with input matrix X of any size
numberOfColumnsInX_norm = size(X_norm,2);
for i = 1:numberOfColumnsInX_norm
	meanOfCurrentFeatureInX = mean(X(:, i));
	mu(:, i) = meanOfCurrentFeatureInX;

	X_norm(:, i) = X_norm(:, i) - mu(:, i);

	standardDeviationOfCurrentFeatureInX = std(X(:, i));
	sigma(:, i) = standardDeviationOfCurrentFeatureInX;

	X_norm(:, i) = X_norm(:, i) ./ sigma(:, i);
end
% ============================================================

end
