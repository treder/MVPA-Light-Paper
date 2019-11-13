function [X_train, y_train, X_test, y_test] = split_into_train_and_test_set(X, y, p)
% Randomly splits inputs X and y into a train set and a test set. 
% p is the fraction of hold out data (default 0.2)

if nargin<3, p=0.2; end

N = length(y);
N_test = round(p*N);

rand_ix = randperm(N);
rand_ix_test = rand_ix(1:N_test);
rand_ix_train = rand_ix(N_test+1:end);

X_train = X(rand_ix_train, :, :, :, :);
X_test  = X(rand_ix_test, :, :, :, :);

y_train = y(rand_ix_train);
y_test  = y(rand_ix_test);