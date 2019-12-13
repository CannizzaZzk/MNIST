function [dists,neighbors] = top_K_neighbors( X_train,y_train,X_test,K )
% 
%   Input: 
%   X_test the test vector with n *1
%   X_train and y_train are the train data set
%   K is the K neighbor parameter

[~, N_train] = size(X_train);
% cal column num of train set, also num of samples
m_train = size(X_train,2);


test_matrix = repmat(X_test,1,m_train);
distance = (X_train - test_matrix) .^ 2;
%distance --> n * m_train matrix

disSum = sum(distance);
%disSum --> 1 * m_train matrix

[dists, neighbors] = sort(disSum);

dists = dists(1:K);
neighbors = neighbors(1:K);

end