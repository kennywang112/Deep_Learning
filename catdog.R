library(keras)
train_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set')
test_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/test_set/test_set')
train_cat<-file.path(train_dir, 'cats')
train_dog<-file.path(train_dir, 'dogs')
test_cat<-file.path(test_dir, 'cats')
test_dog<-file.path(test_dir, 'dogs')
model<-keras_model_sequential()%>%
  layer_conv_2d(filter=32,kernel_size = c(3,3),activation='relu',
                input_shape = c(150,150,3))%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 64,kernel_size=c(3,3),activation='relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 128,kernel_size = c(3,3),activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = 128,kernel_size = c(2,2))%>%
  layer_flatten()%>%
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 512,activation = 'relu')%>%
  layer_dense(units = 1,activation = 'sigmoid')
summary(model)
model%>%compile(
  loss='binary_crossentropy',
  optimizer=optimizer_rmsprop(learning_rate =1e-4),
  metrics=c('acc'))
#顯示擴充的圖像
fnames<-list.files(train_cat,full.names = TRUE)
img_path<-fnames[[3]]
img<-image_load(img_path,target_size = c(150,150))
img_array<-image_to_array(img)
img_array<-array_reshape(img_array,c(1,150,150,3))
augmentation_generator<-flow_images_from_data(img_array,generator = datagen,batch_size = 1)
op<-par(mfrow=c(2,2),pty='s',mar=c(1,0,1,0))
for(i in 1:4){
  batch<-generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
#訓練
test_datagen<-image_data_generator(rescale = 1/255)
train_generator<-flow_images_from_directory(
  train_dir,datagen,target_size = c(150,150),batch_size = 40,class_mode = 'binary')
history<-model%>%fit(
  train_generator,steps_per_epoch = 100,epochs = 30,validation_steps = 50)
plot(history)
