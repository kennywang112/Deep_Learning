library(keras)
dataset<-dataset_boston_housing()
c(c(train_data,train_targets),c(test_data,test_targets))%<-%dataset
str(train_data)
str(test_data)
str(train_targets)
#標準化數據
mean<-apply(train_data,2,mean) #2 by column
std<-apply(train_data,2,sd) #sd變異數OR方差
train_data<-scale(train_data,center = mean,scale=std)
test_data<-scale(test_data,center = mean,scale=std)
#模型定義
build_model<-function(){
  model<-keras_model_sequential()%>%
    layer_dense(units=64,activation = 'relu',input_shape = dim(train_data)[[2]])%>%
    layer_dense(units=64,activation = 'relu')%>%
    layer_dense(units=1)
  
  model%>%compile(
    optimizer='rmsprop',
    loss='mse',
    metrics=c('mae')
  )
}
#k折驗證
k<-4
indices<-sample(1:nrow(train_data))
folds<-cut(1:length(indices),breaks = k,labels = FALSE)
num_epochs<-100
all_scores<-c()
for(i in 1:k){
  cat('processing fold #',i,'\n')
  #準備第K部分的驗證數據
  val_indices<-which(folds==i,arr.ind = TRUE)
  val_data<-train_data[val_indices,]
  val_targets<-train_targets[val_indices]
  #準備其他部分的訓練數據
  partial_train_data<-train_data[-val_indices,]
  partial_train_targets<-train_targets[-val_indices]
  model<-build_model()
    model%>%fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0)
    results<-model%>%evaluate(val_data,val_targets,verbose=0)
    #all_scores<-c(all_scores,results$mae)
}
#保存每日驗證日志
num_epochs<-500
all_mae_histories<-NULL
for(i in 1:k){
  cat('processing fold #',i,'\n')
  #準備第K部分的驗證數據
  val_indices<-which(folds==i,arr.ind = TRUE)
  val_data<-train_data[val_indices,]
  val_targets<-train_targets[val_indices]
  #準備其他部分的訓練數據
  partial_train_data<-train_data[-val_indices,]
  partial_train_targets<-train_targets[-val_indices]
  model<-build_model()
  history<-model%>%fit(partial_train_data,partial_train_targets,
                      validation_data=list(val_data,val_targets),
                      epochs=num_epochs,batch_size=1,verbose=0)
  mae_history<-history$metrics$val_mae
  all_mae_histories<-rbind(all_mae_histories,mae_history)
}
#建構連續平均K折驗證分數的歷史資料
average_mae_history<-data.frame(
  epoch=seq(1:ncol(all_mae_histories)),
  validation_mae=apply(all_mae_histories,2,mean)
)
library(ggplot2)
ggplot(average_mae_history,aes(epoch,validation_mae))+geom_line()
ggplot(average_mae_history,aes(epoch,validation_mae))+geom_smooth()

model<-build_model()
model%>%fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
result<-model%>%evaluate(test_data,test_targets)
result
