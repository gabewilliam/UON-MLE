% call the load function
[features_xy,labels] = loader(500);

%N size of your data
function [features_xy,labels] = loader(N)

%LOAD DATA
labels = load('label.csv');
features_x = load('predx_for_classification.csv');
features_y = load('predy_for_classification.csv');

%SHUFFLE DATA
new_indices = randperm(size(labels,1));

sh_features_x = features_x(new_indices,:);
sh_features_y = features_y(new_indices,:);
sh_labels = labels(new_indices,:);

%RESHAPE DATA
features_xy = cat(2, sh_features_x, sh_features_y);

%Cut data down to size N 
features_xy = features_xy(1:N, :);
labels = sh_labels(1:N,1);

end
