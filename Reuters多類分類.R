library(keras)
#多類分類(新聞專線)
reuters<-dataset_reuters(num_words = 10000)
c(c(train_data,train_labels),c(test_data,test_labels))%<-%reuters
#向量化
vectorize_sequences<-function(sequences,dimension=10000){
  results<-matrix(0,nrow=length(sequences),ncol = dimension)
  for(i in 1:length(sequences))
    results[i,sequences[[i]]]<-1
  results
}
x_train<-vectorize_sequences(train_data)
x_test<-vectorize_sequences(test_data)
#標籤向量化(獨熱編碼或稱分類編碼)
to_one_hot<-function(labels,dimension=46){
  results<-matrix(0,nrow=length(labels),ncol=dimension)
  for(i in 1:length(labels))
    results[i,labels[[i]]+1]<-1
  results
}
one_hot_train_labels<-to_one_hot(train_labels)
one_hot_test_labels<-to_one_hot(test_labels)
#模型定義
model<-keras_model_sequential()%>%
  layer_dense(units = 64,activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 64,activation = 'relu')%>%
  layer_dense(units = 46,activation = 'softmax')
#優化損失
model%>%compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=c('accuracy')
)
#驗證集
val_indices<-1:1000
x_val<-x_train[val_indices,]
partial_x_train<-x_train[-val_indices,]
y_val<-one_hot_train_labels[val_indices,]
partial_y_train<-one_hot_train_labels[-val_indices,]
history<-model%>%fit(
  partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=list(x_val,y_val)
)
plot(history)
#重新訓練
model<-keras_model_sequential()%>%
  layer_dense(units = 64,activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 64,activation = 'relu')%>%
  layer_dense(units = 46,activation = 'softmax')
model%>%compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=c('accuracy')
)
history<-model%>%fit(
  partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=list(x_val,y_val)
)
results<-model%>%evaluate(x_test,one_hot_test_labels)
results
#生成對新數據的預測
predictions<-model%>%predict(x_test)
dim(predictions)
sum(predictions[1,])#向量的係數總和
which.max(predictions[1,])#最高概率的類