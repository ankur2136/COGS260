% train_raw = importdata('iris/train');
% test_raw = importdata('iris/test');
% 
% for i=1:length(train_raw)
%     string = strsplit(char(train_raw(i)), ',');
%     for j=1:length(string)
%         cells = strsplit(char(string(j)));
%         train(i,:)= cells;
%         for k=1:length(cells)
%         end
%     end
% end
% 
% for i=1:length(test_raw)
%     string = strsplit(char(test_raw(i)), ',');
%     for j=1:length(string)
%         cells = strsplit(char(string(j)));
%         test(i,:)= cells;
%         for k=1:length(cells)
%         end
%     end
% end

train_feat = zeros (70, 4);
train_label = ones (70,1);

test_feat = zeros (30,4);
test_label = ones (30,1); 

for i=1:70
    for j=1:4
        train_feat(i,j) = str2double(train(i,j));
    end
    if (strcmp(train(i,5), 'Iris-setosa'))
        train_label(i,1) = -1;
    end
end

for i=1:30
    for j=1:4
        test_feat(i,j) = str2double(test(i,j));
    end
    if (strcmp(test(i,5), 'Iris-setosa'))
        test_label(i,1) = -1;
    end
end

%%
figure(1);
scatter(train_feat(:,1), train_feat(:,2), [], train_label(:,1),'filled');
xlabel('Sepal Length');
ylabel('Sepal Width');
figure(2);
scatter(train_feat(:,1), train_feat(:,3), [], train_label(:,1),'filled');
xlabel('Sepal Length');
ylabel('Petal Length');
figure(3);
scatter(train_feat(:,1), train_feat(:,4), [], train_label(:,1),'filled');
xlabel('Sepal Length');
ylabel('Petal Width');
figure(4);
scatter(train_feat(:,2), train_feat(:,3), [], train_label(:,1),'filled');
xlabel('Sepal Width');
ylabel('Petal Length');
figure(5);
scatter(train_feat(:,2), train_feat(:,4), [], train_label(:,1),'filled');
xlabel('Sepal Width');
ylabel('Petal Width');
figure(6);
scatter(train_feat(:,3), train_feat(:,4), [], train_label(:,1),'filled');
xlabel('Petal Length');
ylabel('Petal Width');