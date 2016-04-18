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
for i=1:100
    dist = 100000;
    chosen_value = -1;
    test = test_images(:,i);
    for j=1:60000
        train = train_images(:,j);
        distance_new = norm(train-test);
        
        if (distance_new < dist)
            dist = distance_new;
            chosen_value = train_labels(j);
        end
    end
    
    if (chosen_value == test_labels(i)) 
        number_correct = number_correct +1;
    else 
        confusion_mat(test_labels(i)+1, chosen_value+1) = ...
                   confusion_mat(test_labels(i)+1, chosen_value+1) + 1;     
        index_wrong(i)=chosen_value;
    end
end
toc
disp (number_correct);
