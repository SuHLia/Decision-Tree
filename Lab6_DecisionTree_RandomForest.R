# install.packages("AER")
# install.packages("randomForest")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("rattle")
# install.packages("caret")

library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(randomForest)
library(AER)

# load dataset affairs
data("Affairs")
dataset = Affairs
head(dataset)

# check the type of each attribute and
# output the summary of the dataset
str(dataset)
summary(dataset)

dataset$yearsmarried <- as.numeric(factor(dataset$yearsmarried))
dataset$religiousness <- as.numeric(factor(dataset$religiousness))
dataset$education <- as.numeric(factor(dataset$education))
dataset$occupation <- as.numeric(factor(dataset$occupation))
dataset$rating <- as.numeric(factor(dataset$rating))
dataset$affairs <- as.factor(dataset$affairs)
dataset$gender <- as.numeric(factor(dataset$gender))
dataset$children <- as.numeric(factor(dataset$children))

# save dataset
# write.csv(dataset, file = "affaits_dataset.csv")

# split the dataset into train data and test data
set.seed(100)
Train_nrow = sample(1:nrow(dataset), 0.7*nrow(dataset))

TrainData = dataset[Train_nrow,]
TestData = dataset[-Train_nrow,]

# check the row number of train and test dataset
nrow(TrainData)
nrow(TestData) 

####################################################################################################

# part1: using decision tree to classify dependent variable affairs
colnames(TrainData)

# Create a decision classification Tree with default parameters
# Plot Decision Tree
DT_Model = rpart(formula = affairs ~ ., data = TrainData, method = "class")
fancyRpartPlot(model = DT_Model, main = "Decision Tree Model")
DT_Model$variable.importance

# predicting on test dataset 
DT_Predict = predict(object = DT_Model, newdata = TestData, type= "class")

# checking accuracy using confusion Matrix
confusionMatrix(DT_Predict, factor(TestData$affairs))

################################################################################################

# part2 using random forest model to do classification

# Create a Random Forest model with default parameters
RF_Model <- randomForest(affairs ~ ., data = TrainData, importance=TRUE)
round(randomForest::importance(RF_Model))

# predicting on test dataset
RF_predTest <- predict(RF_Model, TestData, type = "class")

# use confusionMatrix to find the accuracy
confusionMatrix(RF_predTest, TestData$affairs)
