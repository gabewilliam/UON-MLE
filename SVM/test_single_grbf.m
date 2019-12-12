%test single SVM
sigma=100;
C=10;
grbf_evals = zeros(10,3);

svm_results = kfold_grbf_trainer(C, sigma);
grbf_evals(:,:) = svm_results(:,:);