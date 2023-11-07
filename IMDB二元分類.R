library(keras)
#二元分類(正負面)
imdb<-dataset_imdb(num_words = 10000)
c(c(train_data,train_labels),c(test_data,test_labels))%<-%imdb
str(train_data[[1]])
train_labels[[1]]
max(sapply(train_data,max))
#為了更有趣，解碼回到英文單詞
word_index<-dataset_imdb_word_index()
reverse_word_index<-names(word_index)#反轉，整數索引映射到單詞
names(reverse_word_index)<-word_index
decode_review<-sapply(train_data[[1]], function(index){
  word<-if(index>=3)reverse_word_index[[as.character(index-3)]]#索引值偏移3
  if(!is.null(word)) word else '?'})

vectorize_sequences<-function(sequences,dimension=10000){
  results<-matrix(0,nrow=length(sequences),ncol = dimension)
  for(i in 1:length(sequences))
    results[i,sequences[[i]]]<-1
  results}
x_train<-vectorize_sequences(train_data)
x_test<-vectorize_sequences(test_data)
str(x_train[1,])
y_train<-as.numeric(train_labels)
y_test<-as.numeric(test_labels)
#建構網絡
layer_dense(units = 16,activation = 'relu')#傳遞給每個連接層(16)的參數是該層的隱藏單元(表示該空間的維度)數量
output=relu(dot(W,input)+b)#每個具有relu激活得連接層都會實現該張量運算鏈
#模型定義
model<-keras_model_sequential()%>%
  layer_dense(units = 16,kernel_regularizer=regularizer_l2(0.001),activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 16,kernel_regularizer=regularizer_l2(0.001),activation = 'relu')%>%
  layer_dense(units = 1,activation = 'sigmoid')
#或是這個模型
model<-keras_model_sequential()%>%
  layer_dense(units = 16,activation = 'relu',input_shape = c(10000))%>%
  layer_dropout(rate=0.5)%>%#不同於L2正則化
  layer_dense(units = 16,activation = 'relu')%>%
  layer_dropout(rate = 0.5)%>%
  layer_dense(units = 1,activation = 'sigmoid')
#選擇優化器和損失函數
model%>%compile(
  optimizer='rmsprop',loss='binary_crossentropy',metrics=c('accuracy'))
val_indices<-1:10000
x_val<-x_train[val_indices,]
partial_x_train<-x_train[-val_indices,]
y_val<-y_train[val_indices]
partial_y_train<-y_train[-val_indices]
model%>%compile(
  optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=c('accuracy'))
history<-model%>%fit(
  partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=list(x_val,y_val))
str(history)
plot(history)#過擬合
#重新訓練
model<-keras_model_sequential()%>%
  layer_dense(units = 16,activation = 'relu',input_shape = c(10000))%>%
  layer_dense(units = 16,activation = 'relu')%>%
  layer_dense(units = 1,activation = 'sigmoid')
model%>%compile(
  optimizer='rmsprop',loss='binary_crossentropy',metrics=c('accuracy'))
model%>%fit(x_train,y_train,epochs=4,batch_size=512)#第四個準確率最高
result<-model%>%evaluate(x_test,y_test)
#評論為正的可能性
model%>%predict(x_test[1:10,])















