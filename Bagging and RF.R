hr=read.csv("hr.csv")
head(hr)

colSums(is.na(hr))
str(hr)

unique(hr$sales)
#Renaming 'sales' column
names(hr)[names(hr)=="sales"]<-"Department"
unique(hr$Department)

#Data Partition
index=sort(sample(nrow(hr),nrow(hr)*0.8))
train=hr[index,]
test=hr[-index,]

which(colnames(hr)=="left")
X_train=train[,-7]
y_train=train$left

X_test=test[,-7]
y_test=test$left

library(caret)
library(ipred)
set.seed(1234)

#Bagging
control=trainControl(method="cv",number=10)
btree=train(X_train,as.factor(y_train),method="treebag",trControl=control,verbose=F,keepX=TRUE,coob=TRUE)

#Prediction and Accuracy
predb=predict(btree$finalModel,newdata=X_test,type="prob")

library(ROCR)
predicted=prediction(predb[,2],y_test)
auc=performance(predicted,"auc")
auc=unlist(slot(auc,"y.values"))
auc

#OOB Estimate
print(btree$finalModel)

#Feature Importance
bagimp=varImp(btree)
bagimp
plot(bagimp)

#RandomForest
library(randomForest)
control=trainControl(method="cv",number=10)
tune=expand.grid(mtry=c(3,6,9))
rf=train(X_train,as.factor(y_train),method="rf",trControl=control,tuneGrid=tune,verbose=F)
rf

tune1=expand.grid(mtry=3)
rfmod=train(X_train,as.factor(y_train),method="rf",tuneGrid=tune1,verbose=F)

#Prediction and Accuracy
predrf=predict(rfmod$finalModel,newdata=X_test,type="prob")

predicted=prediction(predrf[,2],y_test)
auc=performance(predicted,"auc")
auc=unlist(slot(auc,"y.values"))
auc

#OOB Estimate
print(rfmod$finalModel)

#Feature Importance
rfimp=varImp(rfmod)
rfimp
plot(rfimp)
