%%
train_images = loadMNISTImages('train-images-idx3-ubyte')';
train_labels = loadMNISTLabels('train-labels-idx1-ubyte')';

test_images = loadMNISTImages('t10k-images-idx3-ubyte')';
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte')';

%%
SVMModel = fitcsvm(train_images,train_labels,'KernelFunction','rbf',...
    'KernelScale','auto','IterationLimit' ,1e1);

%%
results = multisvm(train_images(1:100,:), train_labels(1:100), test_images(1:10,:));
