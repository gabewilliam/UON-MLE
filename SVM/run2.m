sigma_grid = linspace(0.1, 100, 100);

for i = 1:100

    svm_results = kfold_grbf_trainer(sigma_grid(i));
    evals(i,:) = mean(svm_results(:,:));
    
end

f = evals;