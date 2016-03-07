function [Y_predict] = rathp_RDA_test(X_test, RDAmodel, numofClass)
% Testing for QDA
%
% EC 503 Learning from Data
% Spring semester, 2016
% by Prakash Ishwar
%
% Assuming D = dimension of data
% Inputs:
% X_test : test data matrix, each row is a test data point
%
% numofClass : number of class 
% Assuming that the classes are labeled  from 1 to numofClass
%
% QDAmodel: the parameters of QDA classifier which has the follwoing fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigmapooled : D*D  covariance 
% QDAmodel.Pi : numofClass *1 vector, Pi(i) = prior probability of class i

% Output:
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:



covar1=RDAmodel.Sigmapooled;
muy1=RDAmodel.Mu;
Prob1=RDAmodel.Pi;

[test_rows,test_col]=size(X_test);




Dist1=zeros(numofClass,test_rows);
for j=1:test_rows
    for i=1:numofClass
    Dist1(i,j)=(muy1(i,:)/(covar1))*(X_test(j,:))'-0.5.*((muy1(i,:))/(covar1)*(muy1(i,:))')+log((Prob1(i,1)));
    end
end

labels=0:1:(numofClass-1);
[val,Y_predict]=max(abs(Dist1));

Y_predict=labels(Y_predict);
Y_predict=(Y_predict)';



end
