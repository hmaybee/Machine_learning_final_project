library(caret)
library(dplyr)

pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")

##remove row numbers
training <- pml.training[,-1]
testing <- pml.testing[,-1]


##We want to determine if the sensor data can be used predict which exercise
##variation was done, therefore, we wll remove the  user, timestamp and window data
excluded_cols <- c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                   "cvtd_timestamp", "new_window", "num_window")
training <- training %>%
  select(-any_of(excluded_cols))


test_final <- testing %>%
  select(-any_of(excluded_cols))

##change divide by zero and blanks to NA
training[training=="#DIV/0!"] <- NA
test_final[test_final=="#DIV/0!"] <- NA
training[training==""] <- NA
test_final[test_final==""] <- NA

##The testing data set provided does not have the outcome variable, therefore, 
##for training and testing the model, we will break the original training dataset
##into a training and testing data sets.
set.seed(1121)
inTrain = createDataPartition(training$classe, p = 3/4, list=FALSE)
training = training[inTrain,]
testing = training[-inTrain,]


##There are a lot of NA's in the data set. Let's remove columns with more than
##80% NA's
remove <- names(training[,which(colMeans(is.na(training))>=.8)])
training <- training %>%
  select(-any_of(remove))

##Convert classe to factor

training <- training %>%
  mutate(classe = factor(classe))

testing <- testing %>%
  mutate(classe = factor(classe))

##Check for near zero variance
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv ##No TRUE values

dim(training)
##look at correlations of predictors:
descrCor <-  cor(training[,-53])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
##there are 8 predictors that meet this threshhold, try removing them
training <- training[,-highlyCorDescr]

##check if training data is balanced
table(training$classe)
table(testing$classe)
##try a linear svm

trctrl <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(1122)
linear_svm <- train(classe~., data=training, method="svmLinear",
                 trControl=trctrl, preProcess=c("center", "scale"),
                 tuneLength=10)

linear_svm
pred1 <- predict(linear_svm, newdata = testing)
confusionMatrix(pred1, testing$classe)

##try non-linear SVM
set.seed(1122)
non_linear_svm <- train(classe~., data=training, method="svmRadial",
                    trControl=trctrl, preProcess=c("center", "scale"),
                    tuneLength=10)
non_linear_svm


pred2 <- predict(non_linear_svm, newdata = testing)
confusionMatrix(pred2, testing$classe)

test_new <- read.csv("pml-testing.csv")
pred2 <- predict(non_linear_svm, newdata = test_new)
pred2
