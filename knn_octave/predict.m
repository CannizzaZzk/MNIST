function result = predict(neighbor_labels,class_num )


% K_labels --> 1 * K_labels
K = size(neighbor_labels,2);
count = zeros(1,10);

for i = 1 : K
    index = neighbor_labels(i) + 1;
    count(index) = count(index) + 1;
end

[~,result] = max(count);
result = result - 1;

end
