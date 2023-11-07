library(keras)
train_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set')
test_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/test_set/test_set')
conv_base<-application_vgg16(weights = 'imagenet',include_top = FALSE,input_shape = c(150,150,3))
model<-keras_model_sequential()%>%
  conv_base%>%layer_flatten()%>%layer_dense(unit=256,activation='relu')%>%layer_dense(units = 1,activation='sigmoid')
model
cat('before freezing:',length(model$trainable_weights),'\n')
freeze_weights(conv_base)
cat('after freezing:',length(model$trainable_weights),'\n')
train_datagen=image_data_generator(
  rescale = 1/255,rotation_range = 40,width_shift_range = 0.2,height_shift_range = 0.2,
  shear_range = 0.2,zoom_range = 0.2,horizontal_flip = TRUE,fill_mode = 'nearest'
)
test_datagen<-image_data_generator(rescale = 1/255)
train_generator<-flow_images_from_directory(
  train_dir,train_datagen,target_size = c(150,150),
  batch_size = 20,class_mode = 'binary'
)
model%>%compile(
  loss='binary_crossentropy',optimizer=optimizer_rmsprop(learning_rate = 2e-5),metrics=c('accuracy')
)
history<-model%>%fit(
  train_generator,steps_per_epoch=100,epochs=30
)
