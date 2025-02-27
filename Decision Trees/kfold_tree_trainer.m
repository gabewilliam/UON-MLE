
load_data;

%number of folds to use
n = 10;

evaluations = zeros(n,3);

for i = 1:n
   
    train_data =  cell2mat(features_xy_flds(1:end ~= i));
    train_labels = cell2mat(labels_flds(1:end ~= i));
    test_data = cell2mat(features_xy_flds(i));
    test_labels = cell2mat(labels_flds(i));
    
    disp(size(train_data));
    
    %AU1 
    train_labels = train_labels(:,1);
    test_labels = test_labels(:,1);
    
    dTrees(i) = createDecisionTree(train_data, train_labels);
    
    [k,~] = size(test_labels);
    
    %Calculate evaluative measures
    tree_outputs = zeros(k,1);
    
    for j = 1:k
        tree_outputs(j)=classify(test_data(j,:),dTrees(i));
    end
    
    [precision,recall,F1] = evaluation(tree_outputs,test_labels);
    evaluations(i,1) = precision;
    evaluations(i,2) = recall;
    evaluations(i,3) = F1;
    
end

%Save trees for later use
save('trees', 'dTrees');
save('eval', 'evaluations');
