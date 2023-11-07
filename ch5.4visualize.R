model<-application_vgg16(weights = 'imagenet')
train_cats_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set/cats')
fnames<-list.files(train_cats_dir,full.names = TRUE)
img_path<-fnames[[2]]
img<-image_load(img_path,target_size = c(224,224))%>%
  image_to_array()%>%
  array_reshape(dim=c(1,224,224,3))
img_tensor<-img/255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))
layer_outputs<-lapply(model$layers[2:4],function(layer) layer$output)
activation_model<-keras_model(inputs = model$input,outputs = layer_outputs)
activations<-activation_model%>%predict(img_tensor)
layer1_activation<-activations[[1]]
dim(layer1_activation)
plot_channel<-function(channel){
  rotate<-function(x) t(apply(x,2,rev))
  image(rotate(channel),axes=FALSE,asp=1,col=terrain.colors(12))
}
plot_channel(layer1_activation[1,2])

library(keras)
library(tensorflow)
library(magick)
library(viridis)
tf$compat$v1$disable_eager_execution()
model<-application_vgg16(weights = 'imagenet')
img_path<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set/cats/cat.121.jpg')
img<-image_load(img_path,target_size = c(224,224))%>%
  image_to_array()%>%
  array_reshape(dim=c(1,224,224,3))%>%
  imagenet_preprocess_input()
preds <- model %>% predict(img)
max=which.max(preds[1,])
cat_output<-model$output[,max]
last_conv_layer<-model%>%get_layer('block5_conv3')
grads<-k_gradients(cat_output,last_conv_layer$output)[[1]]
pooled_grads<-k_mean(grads,axis=c(1,2,3))
iterate<-k_function(list(model$input),list(pooled_grads,last_conv_layer$output[1,,,]))
c(pooled_grads_value,conv_layer_output_value)%<-%iterate(list(img))
for(i in 1:512){
  conv_layer_output_value[,,i]<-conv_layer_output_value[,,i]*pooled_grads_value[[i]]
}
heatmap<-apply(conv_layer_output_value,c(1,2),mean)
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width=224, height=224 ,bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap ,'img_heatmap.png')
image <- image_read(img_path)
info<-image_info(image)
geometry<-sprintf('%dx%d!',info$width,info$height)
pal<-col2rgb(viridis(20),alpha = TRUE)
alpha<-floor(seq(0,255,length=ncol(pal)))
pal_col<-rgb(t(pal),alpha = alpha,maxColorValue = 255)
write_heatmap(heatmap,'img_overlay.png',width = 14,height = 14,bg=NA,col=pal_col)
image_read('img_overlay.png')%>%image_resize(geometry,filter = 'quadratic')%>%
  image_composite(image,operator = 'blend',compose_args = '30')%>%plot()

