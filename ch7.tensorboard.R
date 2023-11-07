library(keras)
dir.create('my_log_dir')
tensorboard('my_log_dir')
max_feature<-2000
max_len<-500
imdb<-dataset_imdb(num_words = max_feature)
c(c(x_train,y_train),c(x_test,y_test))%<-%imdb
x_train<-pad_sequences(x_train,maxlen = max_len)
x_test<-pad_sequences(x_test,maxlen = max_len)
model<-keras_model_sequential()%>%
  layer_embedding(input_dim = max_feature,output_dim = 128,input_length = max_len,name='embed')%>%
  layer_conv_1d(filters = 32,kernel_size=7,activation='relu')%>%
  layer_max_pooling_1d(pool_size = 5)%>%
  layer_conv_1d(filters = 32,kernel_size=7,activation='relu')%>%
  layer_global_max_pooling_1d()%>%
  layer_dense(units = 1)
model%>%compile(
  optimizer='rmsprop',loss='binary_crossentropy',metrics=c('acc')
)
callbacks=list(callback_tensorboard(
  log_dir = 'my_log_dir',histogram_freq = 1
  )
)
history<-model%>%fit(
  x_train,y_train,
  epochs=20,batch_size=128,validation_split=0.2,callbacks=callbacks
)
