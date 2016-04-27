%% loading

train_images = loadMNISTImages('../train-images-idx3-ubyte')';
train_labels = loadMNISTLabels('../train-labels-idx1-ubyte')';

test_images = loadMNISTImages('../t10k-images-idx3-ubyte')';
test_labels = loadMNISTLabels('../t10k-labels-idx1-ubyte');


%% conversion

count = zeros(10,1);
for i=1:10000
    im = reshape(test_images(i,:),28,28);
    count(test_labels(i)+1) = count(test_labels(i)+1)+1;
    dest = sprintf('test_images_full/%d_%d.jpg', count(test_labels(i)+1), test_labels(i));
    imwrite(im, dest);
end