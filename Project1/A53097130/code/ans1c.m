clc;
close all;

input1 = imread('../Data/snp/1.jpg');
input2 = imread('../Data/snp/2.jpg');
input3 = imread('../Data/snp/3.jpg');
input4 = imread('../Data/snp/4.jpg');
input5 = imread('../Data/snp/5.jpg');

arr = cat(3, input1, input2, input3, input4, input5);
filterSize = [14 3 5 20 7];

for i=1:5    %image counter
    imR  = arr(:,:,(i-1)*3+1);
    imG  = arr(:,:,(i-1)*3+2);
    imB  = arr(:,:,(i-1)*3+3);
    
    resR = medfilt2(imR, [filterSize(i) filterSize(i)]);
    resG = medfilt2(imG, [filterSize(i) filterSize(i)]);
    resB = medfilt2(imB, [filterSize(i) filterSize(i)]);
    
    res = cat(3,resR, resG, resB);
    dest = sprintf('../Results/ans1c_image%d_median_filterSize3x3.jpg', i);
    imwrite(res, dest);
end