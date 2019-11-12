%LOAD DATA

labels = load('label.csv');

features_x = load('predx_for_classification.csv');
features_y = load('predx_for_classification.csv');

labels_au1 = labels(:,1);
labels_au2 = labels(:,2);
labels_au3 = labels(:,3);
labels_au4 = labels(:,4);
labels_au5 = labels(:,5);

features_xy = cat(2, features_x, features_y);

