#Read data from csv files
pml.training <- read.csv("C:/Users/Ankit/Desktop/PML/data/pml-training.csv")
pml.testing <- read.csv("C:/Users/Ankit/Desktop/PML/data/pml-testing.csv")

#Replace all missing values with NA
pml.training[pml.training == "#DIV/0!"] <- NA
pml.training[pml.training == " "] <- NA

#Remove columns with less than 5% data
train <- pml.training[,sapply(pml.training, function(x){sum(is.na(x)==TRUE)}) < nrow(pml.training)*.05]

#Remove unwanted columns
train <- train[,-c(1,2,5)]

#Extract weekday and hour of day from data and remove timestamp
require(lubridate)
train[, 1] <- as.POSIXct(train[, 1], origin = "1970-01-01", tz = "UTC")
train$wday <- wday(train[, 1])
train$hour <- hour(train[, 1])
train<-train[, -1]

#Make non-numeric columns as numeric for PCA
train[,2]<-as.numeric(train[,2])

#create training and test data
inTrain <- createDataPartition(train$classe,p=0.7,list=FALSE)
train_data <- train[inTrain,]
test_data <- train[-inTrain,]

#Principal Component Analysis
pca <- preProcess(train_data[,-56], method = "pca", thresh = .9)
trainPC <- predict(pca, train_data[,-56])
testPC <- predict(pca, test_data[,-56])

#Apply Random Forests Model to predict test data values
require(caret)
rf_model <- train(train_data$classe~.,data=trainPC,method="rf")

#Cross Validation test
confusionMatrix(test_data$classe,predict(rf_model,testPC))
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 1664    1    5    3    1
# B   12 1114   12    0    1
# C    2    8 1003   12    1
# D    4    1   36  922    1
# E    1    5    6    6 1064
# 
# Overall Statistics
# 
# Accuracy : 0.9799         
# 95% CI : (0.976, 0.9834)
# No Information Rate : 0.286          
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.9746         
# Mcnemar's Test P-Value : 0.0001614      
# 
# Statistics by Class:
# 
# Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9887   0.9867   0.9444   0.9777   0.9963
# Specificity            0.9976   0.9947   0.9952   0.9915   0.9963
# Pos Pred Value         0.9940   0.9781   0.9776   0.9564   0.9834
# Neg Pred Value         0.9955   0.9968   0.9879   0.9957   0.9992
# Prevalence             0.2860   0.1918   0.1805   0.1602   0.1815
# Detection Rate         0.2828   0.1893   0.1704   0.1567   0.1808
# Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
# Balanced Accuracy      0.9932   0.9907   0.9698   0.9846   0.9963
#97.99 % accuracy means 2.01 % Out of Sample error

#predict test set
test <- pml.testing[,sapply(pml.testing, function(x){sum(is.na(x)==TRUE)}) < nrow(pml.testing)*.05]
test <- test[,-c(1,2,5)]
testsetPC <- predict(pca, test[,-56])
pred_test <- predict(rf_model,testsetPC)
pred_test
# [1] B A A A A E D B A A B C B A E E A B B B
# Levels: A B C D E