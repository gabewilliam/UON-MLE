function evaluations = outer_kfold_trainer(k,features_xy,labels)

features_xy_flds = kdivide(features_xy, k);
labels_flds = kdivide(labels, k);

evaluations = zeros(k,3);

for i = 1:k
    
    fprintf('Fold %i of %i\n',i,k);
   
    train_data =  cell2mat(features_xy_flds(1:end ~= i));
    train_labels = cell2mat(labels_flds(1:end ~= i));
    test_data = cell2mat(features_xy_flds(i));
    test_labels = cell2mat(labels_flds(i));
    
    %AU1 
    train_labels = train_labels(:,1);
    test_labels = test_labels(:,1);
    
    % grid search on the inner folds
    [q,C] = grid_search(train_data, train_labels,k);
      
    [k,~] = size(test_labels);
    
    %train
    sv = fitcsvm(train_data,train_labels, 'KernelFunction','polynomial', 'PolynomialOrder',q,'BoxConstraint',C);
    sv.predict(test_data);
    
    %Calculate evaluative measures
    %svm_outputs = zeros(k,1);
    sv_predictions = sv.predict(test_data);

    
    [precision,recall,F1] = evaluation(sv_predictions,test_labels);
    evaluations(i,1) = precision;
    evaluations(i,2) = recall;
    evaluations(i,3) = F1;
   
    
end

save('eval.mat', 'evaluations');

end