function cell = kdivide(X,N)
    
    %Total number of data points to be split
    num_data_pts = size(X,1);
    
    %The number of data pts in each fold
    fold_size = floor(num_data_pts/N);
    
    %Data pts left over after division
    remainder = rem(num_data_pts,N);
    
    rowDist = zeros(N,1) + fold_size;
    %Add remainder of data to last fold
    rowDist(N,1) = rowDist(N,1) + remainder;
    cell = mat2cell(X,rowDist);
    
end