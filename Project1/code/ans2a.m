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
    
    fg   = figure;
    set(fg, 'Color', [1 1 1], 'Position', [1 1 1024 768], 'Visible', 'on', 'menubar', 'none');
    [countR,r] = imhist(imR);
    [countG,g] = imhist(imG);
    [countB,b] = imhist(imB);
    subplot(5,2,[1,2,3,4]);
    imshow(arr(:,:,(i-1)*3+1:i*3));
    title('(a) Original Image');
    subplot(5,2,[5,6]);
    stem(r,countR);
    title('(b) Histogram for color Intensity (R)');

    subplot(5,2,[7,8]);
    stem(g,countG);
    title('(c) Histogram for color Intensity (G)');

    subplot(5,2,[9,10]);
    stem(b,countB);
    title('(d) Histogram for color Intensity (B)');

    dest = sprintf('../Results/ans2a_%d.jpg', i);
    saveas(fg, dest);    
end

close all;