library(eeptools)
library(tidyverse)
library(readxl)
library(sqldf)
library(lubridate)
library(hms)
library(caret)
library(dplyr)
library(fastDummies)
library(dummies)
library(car)
library(caret)
library(mice)
library(stringi)
library(olsrr)
library(MASS)
library(zoo)
library(PerformanceAnalytics)
library(Metrics)
library(ggplot2)
library(leaps)
library(glmnet)
library(stats)

setwd("C:\\Lorraine's Program\\Queen's GMMA\\867 Predictive Modeling\\IndividualAssignment1\\restaurant rev")

#############################
### Step 1 import data
############################
res_data<-read.csv('train.csv',header=TRUE) 
res_data_Kaggle<-read.csv('test.csv',header=TRUE) 
summary(res_data)
head(res_data)
summary(res_data_Kaggle)
head(res_data_Kaggle)

################################
### Step 2) Feature Engineer
#################################
#add 1 to columns with 0 value; for log () transformation
res_data[,6:42]<-res_data[,6:42]+1
res_data_Kaggle[,6:42]<-res_data_Kaggle[,6:42]+1
head(res_data_Kaggle)
str(res_data_Kaggle)
dim(res_data_Kaggle)

#create month & week features
res_data$Open.Date<-as.Date(res_data$Open.Date, "%m/%d/%Y")
res_data$Open.Month<-month(res_data$Open.Date)
res_data$Open.Week<-week(res_data$Open.Date)

res_data_Kaggle$Open.Date<-as.Date(res_data_Kaggle$Open.Date, "%m/%d/%Y")
res_data_Kaggle$Open.Month<-month(res_data_Kaggle$Open.Date)
res_data_Kaggle$Open.Week<-week(res_data_Kaggle$Open.Date)

#re-create the Type, to account for additional levels in testing dataset
res_data$Type <-ifelse(res_data$Type=="DT", "Other", res_data$Type)
res_data$Type <-ifelse(res_data$Type=="MB", "Other", res_data$Type)

res_data_Kaggle$Type <-ifelse(res_data_Kaggle$Type=="DT", "Other", res_data_Kaggle$Type)
res_data_Kaggle$Type <-ifelse(res_data_Kaggle$Type=="MB", "Other", res_data_Kaggle$Type)


############################
### Step 3) Create Test and Train
###########################

#train<-subset(res_data, Id<95)
#test<-subset(res_data, (Id>=96 & Id<=139)) 
train<-subset(res_data, Id<110)
test<-subset(res_data, (Id>=111 & Id<=139)) 
str(train)
head(train)
#train


############################
### Step 4) Regularization Begins here
###########################


# create X and y variable
y<-log(train$revenue)
X<- model.matrix(Id~
       
        City.Group*Type+ 
        Open.Month+
        Open.Week*Type+
        

        log(P1)+
        log(P2)+
        log(P3)+
        log(P4)+
        log(P5)+
        log(P6)+
        log(P7)+
        log(P8)+
        log(P9)+
        log(P10)+
        log(P11)+
        log(P12)+
        log(P13)+
        log(P14)+
        log(P15)+
        log(P16)+
        log(P17)+
        log(P18)+
        log(P19)+
        log(P20)+
        log(P21)+
        log(P22)+
        log(P23)+
        log(P24)+
        log(P25)+
        log(P26)+
        log(P27)+
        log(P28)+
        log(P29)+
        log(P30)+
        log(P31)+
        log(P32)+
        log(P33)+
        log(P34)+
        log(P35)+
        log(P36)+
        log(P37)
        ,
        res_data)[,-1]
X<-cbind(res_data$Id,X)


#############
##  Create X  TEST and TRAIN
############


X.training<-subset(X, X[,1]<110)
X.testing<-subset(X, (X[,1]>=111 & X[,1]<=139))



################################
#####  Step 4.1)  LASSO regression; auto Lambda ; alpha=1
#################################
#alpha=1
lasso.fit<-glmnet(x = X.training, y = y, alpha = 1) 
par(mfrow=c(3,3)) # 1 graph plotting
plot(lasso.fit, xvar = "lambda")

# choose the best penalty option
set.seed(123)
crossv <- cv.glmnet(x = X.training, y = y, alpha = 1)  #create 10-fold cross-validation
crossv #displays 2 lowest lambda values with corresponding stats
plot(crossv) #plot MSE values for different values of (log)lambda 


penalty.lasso <- crossv$lambda.min  #optimal penalty parameter, lambda
print(log(penalty.lasso)) #print optimal log(lambda) value

####################
#run the LASSO regression model with the optimal lambda
####################
lasso.opt.fit <-glmnet(x = X.training, y = y, alpha = 1, lambda = penalty.lasso)  

coef(lasso.opt.fit) #display model coefficients 

#exp() converts variables back to the original units
lasso.testing <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.testing)) 
lasso.testing
#calculate and display RMSE  
RMSE.lasso.log.i<-rmse(test$revenue,lasso.testing)
RMSE.lasso.log.i #2184275 with 111 train

########################
####### Step 4.2)  RUn Ridge Regression  alpha = 0
########################
# Ridge
# create X and y variable
y<-log(train$revenue)
X<- model.matrix(Id~
         City.Group*Type+ 
        Open.Month+
        Open.Week*Type+
        
        log(P1)+
        log(P2)+
        log(P3)+
        log(P4)+
        log(P5)+
        log(P6)+
        log(P7)+
        log(P8)+
        log(P9)+
        log(P10)+
        log(P11)+
        log(P12)+
        log(P13)+
        log(P14)+
        log(P15)+
        log(P16)+
        log(P17)+
        log(P18)+
        log(P19)+
        log(P20)+
        log(P21)+
        log(P22)+
        log(P23)+
        log(P24)+
        log(P25)+
        log(P26)+
        log(P27)+
        log(P28)+
        log(P29)+
        log(P30)+
        log(P31)+
        log(P32)+
        log(P33)+
        log(P34)+
        log(P35)+
        log(P36)+
        log(P37)
        ,
        res_data)[,-1]
X<-cbind(res_data$Id,X)

#alpha=0
ridge.fit<-glmnet(x = X.training, y = y, alpha = 0) 
par(mfrow=c(3,3)) # 1 graph plotting
plot(ridge.fit, xvar = "lambda")
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 0)
plot(crossval)
crossval

### display the best penalty lambda for RIDGE regression
penalty.ridge <- crossval$lambda.min 
log(penalty.ridge) #log(lambda) value in the ridge regression
exp(log(penalty.ridge)) 


### estimate the regression model with the optimal penalty###

ridge.opt.fit <-glmnet(x = X.training, y = y, alpha = 0, lambda = penalty.ridge) 
coef(ridge.opt.fit)
ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X.testing))

# RMSE
RMSE.ridge.log.i<-rmse(test$revenue,ridge.testing)
RMSE.ridge.log.i #2209704 with 111 train
 
RMSE.lasso.log.i<-rmse(test$revenue,lasso.testing)
RMSE.lasso.log.i #2184275;  

###############################
######  step 5)   CREATE  kaggle csv
############################


# create X variable for Kaggle's testing dataset

X_kaggle <-model.matrix(Id~  
       City.Group*Type+ 
        Open.Month+
        Open.Week*Type+
        

        log(P1)+
        log(P2)+
        log(P3)+
        log(P4)+
        log(P5)+
        log(P6)+
        log(P7)+
        log(P8)+
        log(P9)+
        log(P10)+
        log(P11)+
        log(P12)+
        log(P13)+
        log(P14)+
        log(P15)+
        log(P16)+
        log(P17)+
        log(P18)+
        log(P19)+
        log(P20)+
        log(P21)+
        log(P22)+
        log(P23)+
        log(P24)+
        log(P25)+
        log(P26)+
        log(P27)+
        log(P28)+
        log(P29)+
        log(P30)+
        log(P31)+
        log(P32)+
        log(P33)+
        log(P34)+
        log(P35)+
        log(P36)+
        log(P37)
        ,
        res_data_Kaggle)[,-1]
X_kaggle<-cbind(res_data_Kaggle$Id,X_kaggle)
head(X_kaggle)
summary(X_kaggle)
str(X_kaggle)


###################################
## FINAL STEP :  transform to normal values exp();  then export data
###################################
#need to add column names and Id 0 to 1  before kaggle submission

kaggle.lasso<- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X_kaggle))
write.csv(kaggle.lasso, file = "PredictedRevenue_lasso_i.csv") 

kaggle.ridge<- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X_kaggle))
write.csv(kaggle.ridge, file = "PredictedRevenue_ridge_i.csv") 
