clc;
close all;

imagefiles = dir('../Data/img/*.jpg');      
nfiles = length(imagefiles);    % Number of files found
pattern = '.jpg';
replacement = '';
    
for ii=1:nfiles
    currentfilename = imagefiles(ii).name;
    currentfilename_stripped = regexprep(currentfilename, pattern, replacement);
    currentimage = imread(sprintf('%s%s','../Data/img/',currentfilename));

    gray = rgb2gray(currentimage);

    sobel = edge(gray, 'Sobel');
    dest = sprintf('../Results/ans3_%s_sobel_th_default.jpg', currentfilename_stripped);
    imwrite(sobel, dest);    
    
    threshold = [5,8,12];    
    for j=1:3
        sobel = edge(gray, 'Sobel', threshold(j)/100);
        dest = sprintf('../Results/ans3_%s_sobel_th_0_%d.jpg', currentfilename_stripped, threshold(j));
        imwrite(sobel, dest);
    end
    
    threshold = [4,5,7];
    canny = edge(gray, 'Canny');
    dest = sprintf('../Results/ans3_%s_canny_th_default.jpg', currentfilename_stripped);
    imwrite(canny, dest);    
    
    for j=1:3
        canny = edge(gray, 'Canny', threshold(j)/10);
        dest = sprintf('../Results/ans3_%s_canny_th_0_%d.jpg', currentfilename_stripped, threshold(j));
        imwrite(canny, dest);
    end
    
    prewitt = edge(gray, 'Prewitt');
    dest = sprintf('../Results/ans3_%s_prewitt_th_default.jpg', currentfilename_stripped);
    imwrite(prewitt, dest);    
    threshold = [5,8,12];    
    for j=1:3
        prewitt = edge(gray, 'Prewitt', threshold(j)/100);
        dest = sprintf('../Results/ans3_%s_prewitt_th_0_%d.jpg', currentfilename_stripped, threshold(j));
        imwrite(prewitt, dest);
    end
end