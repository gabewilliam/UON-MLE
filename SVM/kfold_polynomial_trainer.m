function evaluations = kfold_polynomial_trainer(C,q)

load_data;

%number of folds to use
n = 10;

evaluations = zeros(n,3);

for i = 1:n
    
    fprintf('Fold %i of %i\n',i,n);
   
    train_data =  cell2mat(features_xy_flds(1:end ~= i));
    train_labels = cell2mat(labels_flds(1:end ~= i));
    test_data = cell2mat(features_xy_flds(i));
    test_labels = cell2mat(labels_flds(i));
    
    %disp(size(train_data));
    
    %AU1 
    train_labels = train_labels(:,1);
    test_labels = test_labels(:,1);
    
      
    [k,~] = size(test_labels);
    
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