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

labels_au1 = sh_labels(:,1);
labels_au2 = sh_labels(:,2);
labels_au3 = sh_labels(:,3);
labels_au4 = sh_labels(:,4);
labels_au5 = sh_labels(:,5);

features_xy = cat(2, sh_features_x, sh_features_y);

%Divide data into K folds
K=10;
N = 3000;

%Cut data down to size N
Xtest = sh_features_x(1:N,1);
Ytest = sh_features_y(1:N,1);
test = cat(2, Xtest, Ytest);
 
features_xy = features_xy(1:N, :);
labels = sh_labels(1:N,1);

features_xy_flds = kdivide(features_xy, K);
labels_flds = kdivide(labels, K);