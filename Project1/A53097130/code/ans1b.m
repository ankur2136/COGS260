clc;
close all;

input1 = imread('../Data/img/12084.jpg');
input2 = imread('../Data/img/3096.jpg');
input3 = imread('../Data/img/14037.jpg');

arr = cat(3, input1, input2, input3);
filterSize = [3,5,7];

for i=1:3    %image counter
    im = arr(:,:,(i-1)*3+1:i*3);
    for j=1:3      %filter size
        res = imgaussfilt(im, filterSize(j));
        dest = sprintf('../Results/ans1b%d_%d_gaussion.jpg', i, j);
        imwrite(res, dest);
    end
    
    for j=1:3
        h = fspecial('average', filterSize(j));
        res = imfilter(im, h);
        dest = sprintf('../Results/ans1b%d_%d_average.jpg', i, j);
        imwrite(res, dest);
    end
end