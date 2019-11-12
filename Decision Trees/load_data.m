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

features_xy_flds = kdivide(features_xy, 10);
labels_flds = kdivide(labels,10);