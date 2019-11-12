function [outputArg1] = id3(input)
%input = each column of attribute, e.g. labels_au1
n=0;
p=0;
for i=1:length(input)
    if input(i)==0
        n=n+1;
    else
        p=p+1;
    end
end

answer = -(p/(p+n))*log2(p/(p+n))-(n/(p+n))*log2(n/(p+n));
outputArg1 = answer;

end

