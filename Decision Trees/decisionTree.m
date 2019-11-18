load_data;

features =  cell2mat(features_xy_flds(1));
labels = cell2mat(labels_flds(1));

dTree = createDecisionTree(features,labels);

function tree = createDecisionTree(features, labels)
    
    if all(labels == labels(1))
        
       class = labels(1);
       kids = [];
       op = class;
       attribute = 'null';
       threshold = 'null';
       
    else
        
        [index,Best_Threshold,Best_Gain] = id3(features,labels);
        
        if index == 0
            class = mode(labels);
            kids = [];
            op = class;
            attribute = 'null';
            threshold = 'null';
        else

            attribute = index;
            threshold = Best_Threshold;
            op = attribute + ' <> ' + threshold;

            [m,n1]=size(features); %n1=98
            [~,n2]=size(labels);  %n2=5 || 1

            nextleftdata=[];
            nextleftlabel=[];
            nextrightdata=[];
            nextrightlabel=[];

            for i = 1 : m
                ans = i;
                if features(i,index)<= Best_Threshold
                    nextleftdata=[nextleftdata;features(i,:)];
                    nextleftlabel=[nextleftlabel;labels(i,:)];
                end
                if features(i,index)>Best_Threshold
                    nextrightdata=[nextrightdata;features(i,:)];
                    nextrightlabel=[nextrightlabel;labels(i,:)];
                end
            end

            leftKid = createDecisionTree(nextleftdata, nextleftlabel);
            rightKid = createDecisionTree(nextrightdata, nextrightlabel);

            kids = {leftKid, rightKid};
            
            class = 'null';

        end
        
    end

    tree = createNode(op,kids,class,attribute,threshold);
    
end


function node = createNode(op, kids, class, attribute, threshold)
    
    node.op = op;
    node.kids = kids;
    node.class = class;
    node.attribute = attribute;
    node.threshold = threshold;

end
