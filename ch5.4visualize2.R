library(keras)
library(tensorflow)
library(magick)
library(viridis)
tf$compat$v1$disable_eager_execution()
model<-application_vgg16(weights = 'imagenet')

#sample_images<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set/cats/cat.121.jpg')

files<-file.path('C:/Users/USER/Desktop/Deep Learning/archive/training_set/training_set/')
img_path<-paste(files,list.files(files,recursive=T),sep='/')
set.seed(1)
sample_images<-sample(img_path,10)
sample_images

load<-function(img) {
  img_out<-image_load(img, target_size = c(224, 224)) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, 224, 224, 3))
  return(img_out)
}

grad_cam<-function(model,img_out,y,sample_images){
  preds <- model %>% predict(img_out)
  max=which.max(preds[1,])
  cat_output<-model$output[,max]
  last_conv_layer<-model%>%get_layer('block5_conv3')
  grads<-k_gradients(cat_output,last_conv_layer$output)[[1]]
  pooled_grads<-k_mean(grads,axis=c(1,2,3))
  iterate<-k_function(list(model$input),list(pooled_grads,last_conv_layer$output[1,,,]))
  c(pooled_grads_value,conv_layer_output_value)%<-%iterate(list(img_out))
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
  write_heatmap(heatmap ,paste("image.", y, ".png", sep = ""))
  image <- image_read(sample_images)
  info<-image_info(image)
  geometry<-sprintf('%dx%d!',info$width,info$height)
  pal<-col2rgb(viridis(20),alpha = TRUE)
  alpha<-floor(seq(0,255,length=ncol(pal)))
  pal_col<-rgb(t(pal),alpha = alpha,maxColorValue = 255)
  write_heatmap(heatmap,paste('image.',y,'.png',sep=''),width = 14,height = 14,bg=NA,col=pal_col)
  return(geometry)
}
par(mfrow=c(5,2), mar = c(rep(0.1,4)))
i<-1
while(i<=length(sample_images)){
  image_heat<-load(sample_images[i])
  image_heat<-image_heat/255
  geom_heat<-grad_cam(model,image_heat,i,sample_images[i])
  image_heat<-image_read(paste('image.',i,'.png',sep=''))
  image_heat<-image_resize(image_heat,geom_heat,filter = 'quadratic')
  image_heat<-image_composite(image=image_heat,
                              composite_image = image_convert(image_read(sample_images[i]), colorspace = 'gray')
                              ,operator = 'blend',compose_args = '30')
  plot(image_heat)
  i=i+1
}



