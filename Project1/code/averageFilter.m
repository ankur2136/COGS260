function [ res ] = averageFilter( image, filterSize )
    buffer = (filterSize+1)/2;
    for (i=buffer:image.length-buffer)
        for(j=buffer:image.length-buffer)
            it = image(buffer

end

