#Testing
try(setwd("~/../../Downloads"),silent=TRUE)
try(setwd("~/../Downloads"),silent=TRUE)
library(ggplot2)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(ROCR)
library(e1071)

# Read in data
train <- read.csv("train.csv", stringsAsFactors=FALSE)
holdOutTest  <- read.csv("test.csv",  stringsAsFactors=FALSE)
holdOutTestID<-holdOutTest[,1]

## Adding passenger class as a variable by looking at titles of passengers
train$Title<-sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
train$Title<-gsub(" ", "", train$Title)
#table(train$Title)
train$Title[train$Title %in% c('Capt', 'Don', 'Jonkheer', 'Sir', 'Col', 'Major','Dr', 'Rev')] <- 'Sir'
train$Title[train$Title %in% c('Mlle')] <- 'Miss'
train$Title[train$Title %in% c("Mme", "Ms")] <- 'Mrs'
train$Title[train$Title %in% c("Lady", "theCountess")] <- 'Lady'

###Looking at family size
train$FamilySize<-train$SibSp + train$Parch
train$FamilyType[train$FamilySize == 0 ] <- "Single"
train$FamilyType[train$FamilySize > 0 & train$FamilySize < 3 ] <- "Small"
train$FamilyType[train$FamilySize >= 3 & train$FamilySize < 6 ] <- "Medium"
train$FamilyType[train$FamilySize >= 6 ] <- "Large"
prop.table(table(train$FamilyType,train$Survived),1)


# Function to clean the data
extractFeatures <- function(data,inclSurvived=FALSE) {
    features <- c("Pclass",
                "Age",
                "Sex",
                #"Parch",
                #"SibSp",
                "Fare",
                "Embarked",
                "Title",
                "FamilySize")
  if (inclSurvived==TRUE) {
    fea <- data[,c("Survived",features)]
  } else {
    fea <- data[,features]  
  }
  # fea$Age[is.na(fea$Age)] <- 0
  # fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] <- NA
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  fea<-fea[complete.cases(fea),]
  return(fea)
}

# Clean data
train<-extractFeatures(train,inclSurvived=TRUE)
holdOutTest<-extractFeatures(holdOutTest)

# Partition training data
set.seed(1000)
inTraining<-createDataPartition(train$Survived,p=0.75,list=FALSE)
training<-train[inTraining,]
testing<-train[-inTraining,]

# Train Naives Bayes model
set.seed(1000)
nbFit<-train(factor(Survived)~.,
             data=training,
             method="nb",
             trControl=trainControl(method="repeatedcv",
                                    number=10,
                                    repeats=10))
nbPred<-predict(nbFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],nbPred)
##Accuracy: 0.7528


# Train decision tree model
training$Survived[training$Survived==0]<-"Die"
training$Survived[training$Survived==1]<-"Survived"
testing$Survived[testing$Survived==0]<-"Die"
testing$Survived[testing$Survived==1]<-"Survived"
set.seed(1000)
rpartFit<-train(factor(Survived)~.,
                data=training,
                method="rpart",
                trControl=trainControl(method="repeatedcv",
                                       number=10,
                                       repeats=10,
                                       summaryFunction=twoClassSummary,
                                       classProbs=TRUE),
                metric="ROC")
rpartPred<-predict(rpartFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],rpartPred)
##Accuracy: 0.7978
##Accuracy after adding Title variable: 0.8315
##Accuracy after adding Title and FamilySize variables: 0.8315

# Plot ROC curve
pred = prediction(as.numeric(rpartPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train a boosted tree model
set.seed(1000)
gbmGrid<-expand.grid(interaction.depth=c(1,5,9),
                     n.trees=(1:30)*50,
                     shrinkage=0.1,
                     n.minobsinnode=20)
gbmFit<-train(factor(Survived)~.,
              data=training,
              method="gbm",
              trControl=trainControl(method="repeatedcv",
                                     number=10,
                                     repeats=10),
              verbose = FALSE,
              tuneGrid=gbmGrid,
              summaryFunction=twoClassSummary,
              classProbs=TRUE,
              metric="ROC")
gbmPred<-predict(gbmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],gbmPred)
##Accuracy: 0.7865 
##Accuracy after adding Title variable: 0.8202

# Train random forest model
set.seed(1000)
rfFit<-train(factor(Survived)~.,
             data=training,
             method="rf",
             trControl=trainControl(method="repeatedcv",
                                    number=10,
                                    repeats=10))
rfPred<-predict(rfFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],rfPred)
##Accuracy: 0.7865


# Plot ROC curve
pred = prediction(as.numeric(rfPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')


# Train logistic regression model
set.seed(1000)
glmFit<-train(factor(Survived)~.,
              data=training,
              method="glm",
              trControl=trainControl(method="cv",
                                     number=10))
glmPred<-predict(glmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],glmPred)
##Accuracy: 0.7921
##Accuracy after adding Title variable: 0.8146

# Plot ROC curve
pred = prediction(as.numeric(glmPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train a neural network model
set.seed(1000)
nnFit <- train(factor(Survived)~.,
               data = training, 
               method="nnet",tuneLength=4,maxit=100,trace=F)
nnPred<-predict(nnFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],nnPred)
##Accuracy: 0.7978
##Accuracy after adding Title variable: 0.8202


# Plot ROC curve
pred = prediction(as.numeric(nnPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train SVM Model
set.seed(1000)
svmFit<-svm(factor(Survived)~.,data=training)
svmPred<-predict(svmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],svmPred)
##Accuracy: 0.809


