
load_data;

for i = 1:10
   
    train_data =  cell2mat(features_xy_flds(1:end ~= i));
    train_labels = cell2mat(labels_flds(1:end ~= i));
    test_data = cell2mat(features_xy_flds(i));
    test_labels = cell2mat(labels_flds(i));
    
    %AU1 
    train_labels = train_labels(:,1);
    test_labels = test_labels(:,1);
    
    dTrees(i) = createDecisionTree(train_data, train_labels);
    
    %Calculate evaluative measures
    %Choose best tree
    
end

%Save trees for later use
save('trees', dTrees);
