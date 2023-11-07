library(keras)
train_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set')
test_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/test_set/test_set')
#捲機基
datagen<-image_data_generator(rescale = 1/255)
batch_size<-20
conv_base<-application_vgg16(weights = 'imagenet',include_top = FALSE,input_shape = c(150,150,3))
extract_feature<-function(directory,sample_count){
  features<-array(0,dim=c(sample_count,4,4,512))
  labels<-array(0,dim=c(sample_count))
  generator<-flow_images_from_directory(
    directory = directory,generator = datagen,target_size = c(150,150),
    batch_size = batch_size,class_mode = 'binary'
  )
  i<-0
  while(TRUE){
    batch<-generator_next(generator)
    inputs_batch<-batch[[1]]
    labels_batch<-batch[[2]]
    features_batch<-conv_base%>%predict(inputs_batch)
    index_range<-((i*batch_size)+1):((i+1)*batch_size)
    features[index_range,,,]<-features_batch
    labels[index_range]<-labels_batch
    i<-i+1
    if(i*batch_size>=sample_count)
      break
  }
  list(features=features,labels=labels)
}
train<-extract_feature(train_dir,2000)
test<-extract_feature(test_dir,1000)
reshape_features<-function(features){
  array_reshape(features,dim = c(nrow(features),4*4*512))
}
train$features<-reshape_features(train$features)
test$features<-reshape_features(test$features)
model<-keras_model_sequential()%>%
  layer_dense(units = 256,activation = 'relu',input_shape = 4*4*512)%>%
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 1,activation = 'sigmoid')
model%>%compile(
  loss='binary_crossentropy',
  optimizer=optimizer_rmsprop(learning_rate =2e-5),
  metrics=c('acc'))
history<-model%>%fit(
  train$features,train$labels,epochs=30,batch_size=20
)
