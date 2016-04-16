clc;
close all;
warning('off', 'Images:initSize:adjustingMag');

input1 = imread('../Data/img/12084.jpg');
input2 = imread('../Data/img/3096.jpg');
input3 = imread('../Data/img/14037.jpg');

arr = cat(3, input1, input2, input3);

for i=1:3    %image counter
    im  = arr(:,:,(i-1)*3+1:i*3);
    
    cform2lab = makecform('srgb2lab');
    LAB = applycform(im, cform2lab);
    L = LAB(:,:,1); 
    LAB(:,:,1) = adapthisteq(L,'NumTiles',...
                         [16 16],'ClipLimit',0.005);

    cform2srgb = makecform('lab2srgb');
    J = applycform(LAB, cform2srgb);

    imR = J(:,:,1);
    imG = J(:,:,2);
    imB = J(:,:,3);

    fg   = figure;
    set(fg, 'Color', [1 1 1], 'Position', [1 1 1024 786], 'Visible', 'on', 'menubar', 'none');
    [countR,r] = imhist(imR);
    [countG,g] = imhist(imG);
    [countB,b] = imhist(imB);
    
    subplot(5,4,[1,2,5,6]);
    imshow(im);
    title('(a) Original Image');

    subplot(5,4,[3,4,7,8]);
    imshow(cat(3, imR, imG, imB), 'InitialMagnification', 1000, 'border', 'tight' );
    title('(b) Adaptive Histogram Equalized Image');
    
    subplot(5,4,[9,10,11,12]);
    stem(r,countR);
    title('(c) Histogram of Adaptive Histogram Equalized image for color Intensity (R)');

    subplot(5,4,[13,14,15,16]);
    stem(g,countG);
    title('(d) Histogram of Adaptive Histogram Equalized image for color Intensity (G)');

    subplot(5,4,[17,18,19,20]);
    stem(b,countB);
    title('(e) Histogram of Adaptive Histogram Equalized image for color Intensity (B)');

    dest = sprintf('../Results/ans2c_%d.jpg', i);
    saveas(fg, dest);
    
end