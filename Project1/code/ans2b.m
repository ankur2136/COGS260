clc;
close all;
warning('off', 'Images:initSize:adjustingMag');

input1 = imread('../Data/img/12084.jpg');
input2 = imread('../Data/img/3096.jpg');
input3 = imread('../Data/img/14037.jpg');

arr = cat(3, input1, input2, input3);

for i=1:3    %image counter
    imR  = arr(:,:,(i-1)*3+1);
    imG  = arr(:,:,(i-1)*3+2);
    imB  = arr(:,:,(i-1)*3+3);
    
    [countR,r] = imhist(imR);
    [countG,g] = imhist(imG);
    [countB,b] = imhist(imB);
    
    imR_new = histeq(imR);
    imG_new = histeq(imG);
    imB_new = histeq(imB);

    fg   = figure;
    set(fg, 'Color', [1 1 1], 'Position', [1 1 1024 786], 'Visible', 'on', 'menubar', 'none');
    [countR_new,r_new] = imhist(imR_new);
    [countG_new,g_new] = imhist(imG_new);
    [countB_new,b_new] = imhist(imB_new);
    
    subplot(5,4,[1,2,5,6]);
    imshow(cat(3, imR, imG, imB), 'InitialMagnification', 1000, 'border', 'tight' );
    title('(a) Original Image');
    
    subplot(5,4,[3,4,7,8]);
    imshow(cat(3, imR_new, imG_new, imB_new), 'InitialMagnification', 1000, 'border', 'tight' );
    title('(b) Histogram Equalized Image');
    
    subplot(5,4,[9,10,11,12]);
    stem(r_new,countR_new);
    title('(c) Histogram of Histogram Equalized image for color Intensity (R)');

    subplot(5,4,[13,14,15,16]);
    stem(g_new,countG_new);
    title('(d) Histogram of Histogram Equalized image for color Intensity (G)');

    subplot(5,4,[17,18,19,20]);
    stem(b_new,countB_new);
    title('(e) Histogram of Histogram Equalized image for color Intensity (B)');

    dest = sprintf('../Results/ans2b_%d_new.jpg', i);
    saveas(fg, dest);
    
end