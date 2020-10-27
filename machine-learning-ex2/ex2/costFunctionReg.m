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


h1 = sigmoid(X * theta);
h2 = 1 - sigmoid(X * theta);
l1 = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);
J = (1 / m) * ((- y' * log(h1)) - ((1 - y)' * log(h2))) + l1;


grad1 = (1 / m) * (X(:,1)' * (h1 - y));
gradn = (1 / m) * (X' * (h1 - y)) + ((lambda / m) * theta);
grad = [grad1; gradn(2:end)];


% =============================================================

end
