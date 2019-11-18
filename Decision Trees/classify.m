
function class = classify(features, tree)

while ~isempty(tree.kids)
    left=tree.kids{1};
    right=tree.kids{2};
    if features(tree.attribute)<=tree.threshold
        tree = left;
    else
        tree = right;
    end   
end

class = tree.class;

end

