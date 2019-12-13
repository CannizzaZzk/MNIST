function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, n * m_test matrix.
%           X_train:   training sample features,  n * m matrix.
%           y_train:   training sample labels, 1 * m row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1 * m_test row vector.


m_test = size(X,2);

predicted_label = zeros(1,m_test);
for i = 1 : m_test

    [dists, neighbors] = top_K_neighbors(X_train,y_train,X(:,i),K); 
    % calculate the K nearest neighbors and the distances.

    predicted_label(i) = predict(y_train(neighbors),max(y_train));
    % recognize the label of the test vector.
end

y = predicted_label;

end