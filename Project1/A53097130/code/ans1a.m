clc;
close all;

input = imread('../Data/img/12084.jpg');
image1gray  = rgb2gray(input);
imwrite(input,'../Results/ans1a1.jpg');
imwrite(image1gray,'../Results/ans1a1_gray.jpg');