%%
train_images = loadMNISTImages('train-images-idx3-ubyte')';
train_labels = loadMNISTLabels('train-labels-idx1-ubyte')';

test_images = loadMNISTImages('t10k-images-idx3-ubyte')';
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% [COEFF1, test_images_score]  = pca(test_images);
% [COEFF2, train_images_score] = pca(train_images);

test_images_score = test_images;
train_images_score = train_images;

%%
tic
pca_dim = 784;
Mdl_1 = fitcknn(train_images_score(:,1:pca_dim), train_labels, 'NumNeighbors', 1, 'Distance', 'euclidean');
Mdl_2 = fitcknn(train_images_score(:,1:pca_dim), train_labels, 'NumNeighbors', 1, 'Distance', 'cosine');
Mdl_3 = fitcknn(train_images_score(:,1:pca_dim), train_labels, 'NumNeighbors', 1, 'Distance', 'correlation');
%%
predictions1 = Mdl_1.predict(test_images_score(:,1:pca_dim));
predictions2 = Mdl_2.predict(test_images_score(:,1:pca_dim));
predictions3 = Mdl_3.predict(test_images_score(:,1:pca_dim));

%%
diff1 = predictions1-test_labels;
diff2 = predictions2-test_labels;
diff3 = predictions3-test_labels;
pass1 = length(find(diff1 == 0));
pass2 = length(find(diff2 == 0));
pass3 = length(find(diff3 == 0));
toc

%%
confusion_mat = zeros(10,10);
index_wrong2 = -1*ones(10000, 1);
%%
for i=1:length(diff1)
    if (diff2(i) ~= 0)
        confusion_mat(test_labels(i)+1, predictions2(i)+1) = ... 
            confusion_mat(test_labels(i)+1, predictions2(i)+1) + 1;
        index_wrong2(i) = predictions2(i);
    end
end

%%

index_w = find (index_wrong2 == 7);
for j=1:length(index_w)
    input = reshape(test_images(index_w(j),:), 28, 28);
    dest = sprintf('Results/confused_with_7_%d.jpg', j);
    imwrite(input,dest);
end