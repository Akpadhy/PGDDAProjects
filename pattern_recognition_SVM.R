####################################################################################
########################### Support Vector Machines - Assignment                  ##
########################### Roll Number : DDA1710455                              ##
########################### Batch       : PG Diploma in Data Analytics March 2017 ##
####################################################################################

############################ Handwritten digit recognition problem ##################
# 1. Business Understanding
# 2. Common utility functions used across this code
# 3. Data Understanding and EDA
# 4. Data Preparation
# 5. Model Building 
#   5.1 Linear SVM Model at C=1
#   5.2 Linear SVM Model at C=10
#   5.3 Non Linear model - SVM
# 6  Cross validation 
#   6.1. Tunning linear SVM model 
#   6.2. Tunning Non-linear SVM model 
# 7  Conclusion/Summary 
#####################################################################################
# 1. Business Understanding: 
#----------------------------
# Here the problem is to correctly do recognination of handwritten digit recognition. 
# Data is given in terms of pixel for hand written images for both test and train data.
# Both data set has first column as the target variable rest are the pixel data attributes
# It has 1 categorical variable and 784 numerical attributes

##Loading Neccessary libraries
#----------------------------

library(kernlab)
library(readr)
library(caret)
library(caTools)
library(e1071)
library(ggplot2)
library(gridExtra)

#Loading Data
#----------------------------
mnist_train <- read.csv("mnist_train.csv", header = F, stringsAsFactors = F)
mnist_test <- read.csv("mnist_test.csv", header = F, stringsAsFactors = F)

#####################################################################################
# 2. Common utility functions used across this code

# 2.1 the function prints the statistics of confusion matrix
#-----------------------------------------------------------
printConfusionMatrixStats <- function(confMat){
  total_accuracy_model <- 0
  sum_sensitivity_model <- 0
  sum_specificity_model <- 0
  
  for(i in 1:10){
    #str(cf_mat)
    sum_sensitivity_model <-  sum_sensitivity_model + confMat$byClass[i,1] 
    sum_specificity_model <-  sum_specificity_model + confMat$byClass[i,2]
  }
  avg_sensitivity_model <- sum_sensitivity_model/10
  avg_specificity_model <- sum_specificity_model/10
  total_accuracy_model <- confMat$overall[1]
  cat(sprintf("Total Accuracy of given svm model %f\n", total_accuracy_model))
  cat(sprintf("Average Sensitivity of svm model is %f and Specificity is %f\n", 
              avg_sensitivity_model , avg_specificity_model))
  
}

# 2.2 the function checks if variable exists
#--------------------------------------------
is.defined <- function(sym) {
  sym <- deparse(substitute(sym))
  env <- parent.frame()
  exists(sym, env)
}

#####################################################################################
# 3. Data Understanding and EDA
#----------------------------

#. Note: some part of EDA is covered in the data preparation stage, you can find these plots of EDA by plot1, plot2, plot3 

cat(sprintf("Training set has %d rows and %d columns\n", nrow(mnist_train), ncol(mnist_train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(mnist_test), ncol(mnist_test)))

#Understanding Dimensions
#----------------------------
dim(mnist_train) #60000 observations and 784 variables

#Structure of the dataset
#----------------------------
str(mnist_train) #10000 observations and 784 variables

#printing first few rows
#----------------------------
head(mnist_train)

#Exploring the data
#----------------------------
summary(mnist_train)

#NA values in the train dataset
#----------------------------
sum(is.na(mnist_train)) # Zero null values

#NA values in test dataset
#----------------------------
sum(is.na(mnist_test))  # Zero null values

#Check for duplicate records
nrow(unique(mnist_train))  # there are no duplicate records in training data
nrow(unique(mnist_test))  # there are no duplicate observations in test data as well


#EDA - Exploratory Data analysis
#--------------------------------
# Check the training data and digits available
names(mnist_train)[1] <- "label"
names(mnist_test)[1] <- "label"
plot_digits_train <- ggplot(data = mnist_train, aes(x=mnist_train$label, fill = mnist_train$label)) + geom_bar()
print(plot_digits_train)
# from the above plot, it is inferred that number of observations in each label(digit) categoryare almost same

# Check the test data and digits available
plot_digits_test <- ggplot(data = mnist_test, aes(x=label, fill = label)) + geom_bar()
print(plot_digits_test)

#Now check the intensity of each digit, this is just for EDA purpose, I am not considering it for model preparation
temp_mnist_train <- mnist_train
temp_mnist_train$intensity <- apply(temp_mnist_train[,-1], 1, mean) #takes the mean of each row in train
intbylabel <- aggregate (temp_mnist_train$intensity, by = list(temp_mnist_train$label), FUN = mean)
plot <- ggplot(data=intbylabel, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("average intensity")
#As we can see there are some differences in intensity. The digit "1" is the less intense while the digit "0" is the most intense.

#Now plot the distribution of each digits if it is normal distribution or not
p1 <- qplot(subset(temp_mnist_train, label ==1)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 1")
p2 <- qplot(subset(temp_mnist_train, label ==2)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 2")
p3 <- qplot(subset(temp_mnist_train, label ==3)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 3")
p4 <- qplot(subset(temp_mnist_train, label ==4)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 4")
p5 <- qplot(subset(temp_mnist_train, label ==5)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 5")
p6 <- qplot(subset(temp_mnist_train, label ==6)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 6")
p7 <- qplot(subset(temp_mnist_train, label ==7)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 7")
p8 <- qplot(subset(temp_mnist_train, label ==8)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 8")
p9 <- qplot(subset(temp_mnist_train, label ==9)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 9")

grid.arrange(p1, p2, p3,p4,p5,p6,p7,p8,p9, ncol = 3)

#distributions for 4 and 7 are less "normal" than the distrubution for 1.
#The distribution for 4 looks almost bimodal - perhaps telling that two people write 4 in two ways


#####################################################################################
# 4. Data Preparation
#----------------------------

#. Note: some part of EDA is done here. 

# Changing output variable "lable" to factor type 
label <- as.factor(mnist_train[[1]]) 
test_label <- as.factor(mnist_test[[1]]) 
dftrain.pixel <- mnist_train[,2:ncol(mnist_train)]
dftest.pixel <- mnist_test[,2:ncol(mnist_test)]

# There are so many attributes/variables 784 in number, so here it makes sense to apply the dimensionality reduction or attribute
# reduction using factor rotations in factor analysis.
# Reduce the data set using PCA - principal component analysis technique of factor rotations in factor analysis

# first divide the pixel data set by 255, the reason for doing this is that
# the possible range of pixel is  0 to 255, so dividing the data set by 255
# brings the value of each pixel value to the range of [0-1] which is normalized scale.
Xtrain_reduced <- dftrain.pixel/255
Xtrain_cov <- cov(Xtrain_reduced)
Xtrain_pca<-prcomp(Xtrain_cov)
# tranforming the dataset/ applying PCA to normalized-raw data
# what should be the number of optimum attribtes to be considered, this will come from plot of
# standard deviations data on 2 D plan, here I am selecting 60 because there is a sharp bend at this point 
# indicating beyond this points less significant attributes
plot1 <- plot(Xtrain_pca$sdev)
print(plot1)
Xtrain_final <-as.matrix(Xtrain_reduced) %*% Xtrain_pca$rotation[,1:60]


# Applying PCA to test data
#----------------------------
Xtest_reduced <- dftest.pixel/255
Xtest_final <-as.matrix(Xtest_reduced) %*% Xtrain_pca$rotation[,1:60]

#plots on the reduced data after principal component analysis
plot1 <- plot(Xtrain_pca$sdev)
print(plot1)
plot2 <- plot(Xtrain_pca$x)
print(plot2)
plot3 <- plot(Xtrain_pca$rotation)
print(plot3)
# After seeing the above plots, the distribution of the data set is ellipsoid in nature and the hyper plan equation is of the form
# x^2/a^2 + y^2/b^2 + 1/c^2 = 1

#####################################################################################
# 5. Model Building 
#----------------------------
trainFinal <- cbind.data.frame(label, Xtrain_final)
names(trainFinal)[1] <- "label"
trainFinal$label <- as.factor(trainFinal$label)

testFinal <- cbind.data.frame(test_label, Xtest_final)
names(testFinal)[1] <- "label"
testFinal$label <- as.factor(testFinal$label)

# First I ran all the below models with the entire data set and come up with the stattistics of all models as comments
# For examiner to evaluate my models, I ran the same below models with the sample data

# since the original train and test data contains 60000 and 10000 observations, for better performance
# of the model to run in the minimum time is better to prepare the model with the sample data

#since the data set is huge, take 30% of the data as sampling
indices_train = sample(1:nrow(trainFinal), 0.3*nrow(trainFinal))
indices_test = sample(1:nrow(testFinal), 0.3*nrow(testFinal))

trainFinal = trainFinal[indices_train,]
testFinal = testFinal[indices_test,]

# Impotant note: for each step of my SVM analysis, I prepared two models, one with the entire data set with PCA
# second with  the sampling data set with PCA.

#--------------------------------------------------------------------
# 5.1 Linear model - SVM  at Cost(C) = 1
#####################################################################
# Model with C = 1
#----------------------------

# Start the clock!
ptm <- proc.time()
model_1<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=1)

# Predicting the model results 
evaluate_1<- predict(model_1, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_1 <- confusionMatrix(evaluate_1, testFinal$label, positive = "Yes")

print(conf_matrix_1)



#print the statistics of confusion matrix of model_1
if(is.defined(conf_matrix_1)){
  printConfusionMatrixStats(conf_matrix_1)
}
# Stop the clock
proc.time() - ptm
#-----------------------------------------Statistics - Linear model - SVM  at Cost(C) = 1--------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy                      - 0.979
# Sensitivity                   - 0.979 
# Specificity                   - 0.998
# processing time of the model  - 5.4 minutes
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
  
#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.970
# Sensitivity                   - 0.970 
# Specificity                   - 0.996
# processing time of the model  - 55.28 seconds
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#-------------------------------------------------------------------------------------------------------------------------------
# 5.2 Linear model - SVM  at Cost(C) = 10
#####################################################################
# Model with C = 10
#----------------------------

# Start the clock!
ptm <- proc.time()

model_2<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=10)

# Predicting the model results 
evaluate_2<- predict(model_2, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_2 <- confusionMatrix(evaluate_2, testFinal$label, positive = "Yes")

#print the statistics of confusion matrix of model_2
print(conf_matrix_2)
if(is.defined(conf_matrix_2)){
  printConfusionMatrixStats(conf_matrix_2)
}

# Stop the clock
proc.time() - ptm

#-----------------------------------------Statistics - Linear model - SVM  at Cost(C) = 10-------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy    - 0.984
# Sensitivity - 0.984
# Specificity - 0.998
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.974
# Sensitivity                   - 0.974 
# Specificity                   - 0.997
# processing time of the model  - 46 seconds
# Insight from this model       - with the sample accuracy of 97% and C=10, the model is highly overfitted.
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
#------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------
# 5.3 Non Linear model - SVM
#####################################################################

# RBF kernel 
#----------------------------

# Start the clock!
ptm <- proc.time()
model_rbf <- ksvm(label ~ ., data =trainFinal,scale=FALSE, kernel = "rbfdot")


# Predicting the model results 
Eval_RBF<- predict(model_rbf, testFinal[,-1])

#confusion matrix - RBF Kernel
conf_matrix_model_rbf <- confusionMatrix(Eval_RBF,testFinal$label)
#print the statistics for confuusion matrix
print(conf_matrix_model_rbf)
printConfusionMatrixStats(conf_matrix_model_rbf)

# Stop the clock
proc.time() - ptm

#-----------------------------------------Statistics - Non Linear model - SVM -------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy    - 0.979
# Sensitivity - 0.979
# Specificity - 0.997

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.970
# Sensitivity                   - 0.970 
# Specificity                   - 0.970
# processing time of the model  - 55 seconds
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
#------------------------------------------------------------------------------------------------------------------------------

#####################################################################################
# 6  Cross validation
#####################################################################
# 6.1 Hyperparameter tuning and Cross Validation - Linear - SVM 
#####################################################################

# We will use the train function from caret package to perform crossvalidation

# Start the clock!
ptm <- proc.time()

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of  C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation

fit.svm <- train(label~., data=trainFinal, method="svmLinear", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm)
###############################################################################
# Best tune  C=1, Accuracy - 0.929   for the entire data set i.e.  60000 observations of train data and 10000 observations of test data
# Best tune  C=1, Accuracy -  0.926  with the sample dataset i.e.  18000 observations of train data and 3000 observations of test da
###############################################################################

# Plotting model results
plot(fit.svm)

###############################################################################

# Valdiating the model after cross validation on test data

evaluate_linear_test<- predict(fit.svm, testFinal)
conf_matrix <- confusionMatrix(evaluate_linear_test, testFinal$label)


#print the statistics of confusion matrix 
print("Model using Hyperparameter tuning and Cross Validation - Linear - SVM ")
print(conf_matrix)
printConfusionMatrixStats(conf_matrix)

# Stop the clock
proc.time() - ptm
#-----------------------------------------Statistics - Cross Validation - Linear - SVM ----------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy    - 0.9358
# Sensitivity - 0.934723
# Specificity - 0.992872

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.931
# Sensitivity                   - 0.947 
# Specificity                   - 0.992
# processing time of the model  - 6.19 minutes

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
#------------------------------------------------------------------------------------------------------------------------------

#####################################################################
# 6.2. Hyperparameter tuning and Cross Validation - Non-Linear - SVM 
#####################################################################

# Start the clock!
ptm <- proc.time()

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of "sigma" and C values. 
grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))


# Performing 5-fold cross validation
fit.svm_radial <- train(label~., data=trainFinal, method="svmRadial", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm_radial)
#####################################################################
# Best tune at sigma = 0.03 & C=2, Accuracy - 0.985 with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Best tune at sigma = 0.02 & C=3, Accuracy - 0.976 with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
#####################################################################

# Plotting model results
plot(fit.svm_radial)


######################################################################
# Checking overfitting - Non-Linear - SVM
######################################################################

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm_radial, testFinal)
conf_matrix_non_linear <- confusionMatrix(evaluate_non_linear, testFinal$label)

print(conf_matrix_non_linear)
printConfusionMatrixStats(conf_matrix_non_linear)
# Stop the clock
proc.time() - ptm

#-----------------------------------------Statistics - Cross Validation - Non Linear - SVM ------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#Statistics captured when run with the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
# Accuracy    - 0.985
# Sensitivity - 0.984
# Specificity - 0.998
# processing time of the model  - 20 hours
# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity

#Statistics captured when run with the sample dataset i.e.  18000 observations of train data and 3000 observations of test data
# Accuracy                      - 0.976
# Sensitivity                   - 0.975 
# Specificity                   - 0.997
# processing time of the model  - 4 hours

# Note: Here Sensitivity and Specificity are average because there are 10 classes and
#       each has its own sensitivity and specificity
#------------------------------------------------------------------------------------------------------------------------------

######################################################################
# 7. Conclusion/Summary
######################################################################

# For the enire dataset i.e.  60000 observations of train data and 10000 observations of test data
#--------------------------------------------------------------------------------------------------
#S.NO	"Model Name"	                     "Accuracy"	"Average Sensitivity"	"Average Specificity"	"processing time"	"C value"	    "E value"	"Remarks"
# 1	  linear model  with C = 1             0.979	      0.979	                0.998	               5.4  minutes	      	1	        	 NA	
# 2	  linear model  with C = 10 	         0.984	      0.984	                0.998	               3.2  minutes	      	10	      	     NA	
# 3	  Model Non-Linear - SVM	             0.979	      0.979	                0.997	               4.66 minutes	      	NA     	  	     NA	        
# 4	  Cross Validation - Linear - SVM	     0.936	      0.931	                0.941	               70   minutes	      	1 to 5	  	     NA	          
#     4.1  ----->Best tune  C=1, Accuracy - 0.9296334
# 5	  Cross Validation - Non-Linear - SVM  0.985	      0.984	                0.998	               20   hours	          1 to 5	  	0.1 to 0.5        
#     5.1  ----->Best tune at sigma = 0.03 & C=2, Accuracy - 0.985
  
  # For the enire dataset i.e.  18000 observations of train data and 3000 observations of test data
  # --------------------------------------------------------------------------------------------------
  #S.NO	"Model Name"	                     "Accuracy"	"Average Sensitivity"	"Average Specificity"	"processing time"	"C value"	    "E value"	"Remarks"
  # 1	  linear model  with C = 1             0.970	      0.970	                0.997	               59  seconds	      	1	        	 NA	
  # 2	  linear model  with C = 10 	         0.974	      0.974	                0.997	               46  seconds	      	10	      	     NA	
  # 3	  Model Non-Linear - SVM	             0.972	      0.979	                0.997	               4.66 minutes	      	NA     	  	     NA	        
  # 4	  Cross Validation - Linear - SVM	     0.936	      0.931	                0.992	               6.19 minutes	      	1 to 5	  	     NA	          
  #     4.1  ----->Best tune  C=1, Accuracy - 0.9296334
  # 5	  Cross Validation - Non-Linear - SVM  0.976	      0.975	                0.975	               4   hours	          1 to 5	  	0.01 to 0.05        
  #     5.1  ----->Best tune at sigma = 0.02 & C=3, Accuracy - 0.971	  
  

#After compapring all the SVM models for both linear and Non linear followoing are the conclusions:
#--------------------------------------------------------------------------------------------------
#Note: Here I have run the SVM models for both the entire data set and sampling data set

#############################################################################################################################
# 7.1 -------------------------------FINAL CONCLUSION -----------------------------------------------------------------------
# 7.1.1. For the enire dataset i.e.  60000 observations of train data and 10000 observations of test data, it is confirmed that
#        the accuracy of non-linear model is better than linear model i.e. 0.985 > 0.936.
#        The best tune model is #5 'SVM Non-linear with sigma = 0.03 & C=2, Accuracy - 0.985', and it is also equal
#        to the accuracy of test data which is 0.985  

# 7.1.2. For the sample dataset i.e.  18000 observations of train data and 3000 observations of test data, it is confirmed that
#        the accuracy of non-linear model is better than linear model i.e. 0.976 > 0.936.
#        The best tune model is #5 'SVM Non-linear with sigma = 0.02 & C=3, Accuracy - 0.971', and it is also equal
#        to the accuracy of test data which is 0.975
# 7.1.3. From both the above observations it is confirmed that SVM Non-linear model with accuracy of 0.975  is the best.
#        Now the question arises which sigma and C need to be selected.
#           -->I have selected sigma = 0.02 & C=3, reason is that this model is neither biased nor overfitted.
#           -->At this model, another good thing is that Accuray, average sensitivity and average specificity are equal to 0.975
#           -->Also the non-linearity is very less.
#           -->The proof of non-linearity comes from the plot - plot(Xtrain_pca$x). The polynomial is ellipsoid is nature
#           ------------------------------------------------------------------------------------------------------------------
#           -->So the most optimum model is SVM Non-linear with sigma = 0.02 & C=3, Accuracy - 0.971 approximated to 0.97 i.e. 97%
#               --> This is also confirmed with 0.976 accuracy with the test data i.e. 98%. It also signifies, model is very stable 
#           ------------------------------------------------------------------------------------------------------------------
#   
#------------------------------------------------------------------------------------------------------------------------------
####################################------END OF ASSIGNMENT--------############################################################


