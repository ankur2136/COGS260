%%
train_images = loadMNISTImages('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');

test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

%%
tic
confusion_mat = zeros(10,10);
index_wrong = -1*ones(10000, 1);
number_correct = 0;

for j=1:100
    test_rep = repmat(test_images(:,j)', 60000,1);
    distance_new = (test_rep-train_images').^2;
    distance_sum = sum(distance_new,2);
    
    [m, index] = min(distance_sum);

    if (train_labels(index) == test_labels(j))
        number_correct = number_correct +1;
    else 
        confusion_mat(test_labels(j)+1, train_labels(index)+1) = ...
                   confusion_mat(test_labels(j)+1, train_labels(index)+1) + 1;     
        index_wrong(j)=train_labels(index);
    end
end

toc
disp (number_correct);
%%
clear train_images
clear test_images
clear train_labels
clear test_labels
clear test
clear train
save('workspace.mat')