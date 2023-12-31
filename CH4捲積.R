Sys.setenv(KERAS_BACKEND = "keras")
Sys.setenv(THEANO_FLAGS = "device=gpu,floatX=float32")
library(keras)
model<-keras_model_sequential()%>%
  layer_conv_2d(filters = 32,kernel_size = c(3,3),activation = 'relu',input_shape = c(28,28,1))%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 64,kernel_size = c(3,3),activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 64,kernel_size = c(3,3),activation = 'relu')
model
#添加分類器
model<-model%>%layer_flatten()%>%
  layer_dense(unit=64,activation='relu')%>%
  layer_dense(unit=10,activation = 'softmax')
model
#訓練捲機網絡
mnist<-dataset_mnist()
c(c(train_images,train_labels),c(test_images,test_labels))%<-%mnist

train_images<-array_reshape(train_images,c(60000,28,28,1))
train_images<-train_images/255
test_images<-array_reshape(test_images,c(10000,28,28,1))
test_images<-test_images/255
train_labels<-to_categorical(train_labels)
test_labels<-to_categorical(test_labels)
model%>%compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=c('accuracy')
)
model%>%fit(
  train_images,train_labels,
  epochs=5,batch_size=64
)
#評估模型
results<-model%>%evaluate(test_images,test_labels)
results
