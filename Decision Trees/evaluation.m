

function [precision,recall,F1] = evaluation(tree_results,test_labels)
[TP,FP,TN,FN] = deal(0);


for i=1:length(tree_results) %labels from training
    for j=1:length(test_labels) %labels from test set
        if(tree_results(i,j)==1 && test_labels(i,j)==1)
            TP=TP+1;
        elseif(test_results(i,j)==0 && test_labels(i,j)==0)
            FP=FP+1;    
        elseif(test_results(i,j)==0 && test_labels(i,j)==0)
            TN=TN+1;
        else
            FN=FN+1;
        end
    end
end


precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*((precision*recall)/(precision+recall));

end 

   
           
           
            


