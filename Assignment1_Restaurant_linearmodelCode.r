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


setwd("C:\\Lorraine's Program\\Queen's GMMA\\867 Predictive Modeling\\IndividualAssignment1\\restaurant rev")

########################
### Step 1 import data ###
####################
res_data<-read.csv('train.csv',header=TRUE) 
res_data_Kaggle<-read.csv('test.csv',header=TRUE) 
summary(res_data)
head(res_data)
summary(res_data_Kaggle)
head(res_data_Kaggle)


#####################
### Step 2) Feature Engineer
########################
#add 1 to columns with 0 value
res_data[,6:42]<-res_data[,6:42]+1
res_data_Kaggle[,6:42]<-res_data_Kaggle[,6:42]+1

# Feature Engineer
res_data$Open.Date<-as.Date(res_data$Open.Date, "%m/%d/%Y")
res_data$Open.Month<-month(res_data$Open.Date)
res_data$Open.Week<-week(res_data$Open.Date)

res_data_Kaggle$Open.Date<-as.Date(res_data_Kaggle$Open.Date, "%m/%d/%Y")
res_data_Kaggle$Open.Month<-month(res_data_Kaggle$Open.Date)
res_data_Kaggle$Open.Week<-week(res_data_Kaggle$Open.Date)


res_data$Type <-ifelse(res_data$Type=="DT", "Other", res_data$Type)
res_data$Type <-ifelse(res_data$Type=="MB", "Other", res_data$Type)

res_data_Kaggle$Type <-ifelse(res_data_Kaggle$Type=="DT", "Other", res_data_Kaggle$Type)
res_data_Kaggle$Type <-ifelse(res_data_Kaggle$Type=="MB", "Other", res_data_Kaggle$Type)


# remove date and city column. time series out of consideration
res_data<-subset(res_data,select = -c(Open.Date,City,Id))
res_data_Kaggle<-subset(res_data_Kaggle,select = -c(Open.Date,City,Id))

plot(density(res_data$revenue), main="Density Plot: revenue")


#################
### step 3) Hold Out Validation: Split data into TEST & TRAIN at random
#################
set.seed(123)
sample <- sample.int(n = nrow(res_data), size = floor(.8*nrow(res_data)), replace = F)
train<- res_data[sample, ]
test<- res_data[-sample, ]
str(train)



#################
### step 4) Building Linear and Log-linear model
#################

fit<-lm( 
        revenue~.
        ,
             
        train)

pred.test<-predict(fit,test)

par(mfrow=c(3,3))
plot(fit)
plot(density(resid(fit)))

summary(fit)



################################################
### Try LOG  ###
#############################

fit.log<-lm( 
  log(revenue)~
        City.Group+
        Open.Month+
        Open.Week+
        Type+
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
   , train)

pred.test.log<-exp(predict(fit.log,test))

par(mfrow=c(3,3))
plot(fit.log)
plot(density(resid(fit.log)))

summary(fit.log)


###############################################
#  Try Log-linear with interaction features
#########################
# the city might influence what type of store it is.
fit.log.i<-lm( 
  log(revenue)~
        
        City.Group*Type+
        Open.Week*Type+
        Open.Month+
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
          log(P37), train)

pred.test.log.i<-exp(predict(fit.log.i,test))

par(mfrow=c(3,3))
plot(fit.log.i)
plot(density(resid(fit.log.i))) # NaN 

summary(fit.log.i)

##########################
#####  Evaluation RMSE  ###
#########################

# 
RMSE.lm<-rmse(test$revenue,pred.test)
RMSE.lm
RMSE.log<-rmse(test$revenue,pred.test.log)
RMSE.log
RMSE.log.i<-rmse(test$revenue,pred.test.log.i)
RMSE.log.i

##########################
#####  Final Step
#########################

#this is log with interaction
pred.test.step.log.i.kaggle<-exp(predict(step.fit.log.i, res_data_Kaggle))
write.csv(pred.test.step.log.i.kaggle, file = "PredictedRevenue_log.i.csv") 

#this is log without interaction
pred.test.step.log.kaggle<-exp(predict(step.fit.log, res_data_Kaggle))
write.csv(pred.test.step.log.kaggle, file = "PredictedRevenue_log.csv") 

#this is reglog with interaction
pred.test.fit.log.i.kaggle<-exp(predict(fit.log.i, res_data_Kaggle))
write.csv(pred.test.fit.log.i.kaggle, file = "PredictedRevenue_regularlog.i.csv") 

#this is reg linear model without interaction
pred.test.fit.kaggle<-exp(predict(fit.log, res_data_Kaggle))
write.csv(pred.test.fit.kaggle, file = "PredictedRevenue_regularlm.csv")

head(pred.test.step.log.kaggle)
dim(res_data_Kaggle)





