function [w] = perceptron(X,Y,w_init)
%PERCEPTRON Summary of this function goes here

w = w_init;
for iteration = 1 : 100  %<- in practice, use some stopping criterion!
  for ii = 1 : size(X,2)         %cycle through training set
    if sign(w'*X(:,ii)) ~= Y(ii) %wrong decision?
      w = w + X(:,ii) * Y(ii);   %then add (or subtract) this point to w
    end
  end
%  sum(sign(w'*X)~=Y)/size(X,2);   %show misclassification rate
end

