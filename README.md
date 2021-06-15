# Perform-a-T-test - data science research 
a synchronized ten rounds 10-fold cross validation tests to obtain the classification accuracy scores for the Naive Bayesian and the AdaBoost classifiers learned from the dataset. pecifically, in each round, the same training set should be used to train the classifiers, and the same testing set should be used to measure the classification accuracy scores. Your program should use functions from the SciKit-Learn to create random folds, to learn classifiers, to test the classifiers and and to calculate classification accuracy scores.




The following table contains the outcome of classifying 10 testing tuples using a proba- bilistic classifier. For each tuple, the actual class (P or N) is given in the second column, and the probability (of class P) returned by the classifier is in the third column.
 ![Screen Shot 2021-06-15 at 12 01 43 AM](https://user-images.githubusercontent.com/43187819/121995634-dfcf5100-cd6c-11eb-9b19-c45bfbd5485d.png)

sklearn.naive_bayestolearnaGuassianNaiveBayesclassifier using df3 as the training data, and use the learned predictive model to predict the status of a user provided unseen data, for example,
t =< department : systems, status :?, age : 28, salary : 50K >



OUTPUT: ![Screen Shot 2021-06-15 at 12 02 08 AM](https://user-images.githubusercontent.com/43187819/121995683-fa092f00-cd6c-11eb-8684-88fcce0dc264.png)
!
