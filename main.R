# Setup

library(caret)
library(rpart)
library(kernlab)
library(randomForest)

library(rstudioapi)

rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

train <- read.csv('train.csv', stringsAsFactors = T)
test <- read.csv('test.csv', stringsAsFactors = T)

# Data Checking

names(train)
str(train)

sapply(train, function(x) sum(is.na(x)))
sapply(train, function(x) length(which(x == '')))

# EDA

pairs(train[which(sapply(train, class) == 'numeric')])
hist(train$Age, breaks = 50, main = 'Age')
hist(train$Fare, breaks = 100, main = 'Fare')

# Data Cleaning

## Feature Modifications

train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)

## Imputation: Age

age.train <- train[is.na(train$Age) == F, ]
age.pred <- train[is.na(train$Age) == T, ]
age.grid <- expand.grid(.cp = seq(0.001, 0.02, by = 0.001))
age.fit <- train(Age ~., data = age.train,
                 method = 'rpart',
                 tuneGrid = age.grid,
                 trControl = trainControl(method = 'CV',
                                          number = 5))

train$Age[is.na(train$Age) == T] <- predict(age.fit, newdata = age.pred)

## Imputation: Cabin, Embarked

train$Cabin <- factor(ifelse(train$Cabin == '', 'unknown', 'known'))
train$Embarked[which(train$Embarked == '')] = 'S'

## Feature Engineering: Name to Title

title <- c('Dr', 'Master', 'Mr', 'Mrs', 'Ms', 'Miss')
train$Title <- NA

for (i in 1:6) train$Title[grep(title[i], train$Name)] <- title[i]
train$Title[is.na(train$Title)] <- 'Others'

train$Title <- as.factor(train$Title)

# Modelling

## Features Selection

filtered.model <-
  Survived ~ Pclass + Sex + SibSp + Embarked +
  Age + Parch + Fare + Cabin + Title

null <- glm(Survived ~ 1, family = binomial, data = train)
filtered <- glm(filtered.model, family = binomial, data = train)

step(null, scope= list(lower = null, upper = filtered),
     direction = 'forward')

model <-
  Survived ~ Title + Pclass + SibSp + Age + Parch + Cabin + Sex

##  Decision Trees

rpart.grid <- expand.grid(.cp = seq(0.001, 0.02, by = 0.001))
fit.rpart <- train(model, data = train,
                   method = 'rpart',
                   tuneGrid = rpart.grid,
                   trControl = trainControl(method = 'CV',
                                            number = 5))

fit.rpart$results
max(fit.rpart$results[, 'Accuracy'])

# Random Forest

rf.grid <- expand.grid(.mtry = 2:9)
fit.rf <- train(model, data = train,
                method = 'rf', ntree = 400,
                tuneGrid = rf.grid,
                trControl = trainControl(method = 'CV',
                                         number = 5))

fit.rf$results
max(fit.rf$results[, 'Accuracy'])

## SVM (Kernel: linear)

svm.lin.grid <- expand.grid(.C = c(10, 100))
fit.svm.lin <- train(model, data = train,
                     method = 'svmLinear',
                     tuneGrid = svm.lin.grid,
                     trControl = trainControl(method = 'CV',
                                              number = 5))
fit.svm.lin$results
max(fit.svm.lin$restults)

## SVM (Kernel: polynomial)

svm.pol.grid <- expand.grid(.C = c(0.1, 1, 10),
                            .scale = seq(0, 1, by = .1),
                            .degree = 1:5)
fit.svm.pol <- train(model, data = train,
                     method = 'svmPoly',
                     tuneGrid = svm.pol.grid,
                     trControl = trainControl(method = 'CV',
                                              number = 5))
fit.svm.pol$results
max(fit.svm.pol$results)

## SVM (Kernel: radial)

svm.rad.grid <- expand.grid(.C = c(0.1, 1, 10),
                            .sigma = c(0.1, 1))
fit.svm.rad <- train(model, data = train,
                     method = 'svmRadial',
                     tuneGrid = svm.rad.grid,
                     trControl = trainControl(method = 'CV',
                                              number = 5))

# Evaluation

importance(fit.rpart$finalModel)
varImpPlot(fit.rpart$finalModel)

# Final Prediction

## Test Data Preparation

str(test)
sapply(test, function(x) sum(is.na(x)))

test$Pclass <- as.factor(test$Pclass)
test$Cabin <- factor(ifelse(test$Cabin == '', 'unknown', 'known'))
test$Fare[is.na(test$Fare)] <- mean(train$Fare)

test$Title <- NA
for (i in 1:6) test$Title[grep(title[i], test$Name)] <- title[i]
test$Title[is.na(test$Title)] <- 'Others'
test$Title <- as.factor(test$Title)

age.new.set <- rbind(train[, -2], test)
age.test.fit <- train(Age ~ .,
                      data = age.new.set[is.na(age.new.set$Age) == F, ],
                      method = 'rpart',
                      tuneGrid = age.grid,
                      trControl = trainControl(method = 'CV',
                                               number = 5))
test$Age[is.na(test$Age) == T] <- predict(age.test.fit,
                                          newdata = test[is.na(test$Age) == T,])

## Prediction

pred.rf <- predict(fit.rf, newdata = test)
head(pred.rf)

a <- data.frame(test$PassengerId)
b <- data.frame(pred.rf)
answer <- data.frame(a, b)
names(answer) <- c('PassengerId', 'Survived')

## Export Predictions

write.csv(answer, file = 'submission.csv', row.names = F)
