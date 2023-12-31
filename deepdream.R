library(keras)
library(tensorflow)
library(reticulate)
use_condaenv("r-reticulate", required = TRUE)
tf$compat$v1$disable_eager_execution()

k_set_learning_phase(0)
model<-application_inception_v3(
  weights = 'imagenet',include_top = 'FALSE'
)
layer_contributions<-list(
  mixed1=0.2,mixed3=3
)
loss<-k_variable(0)
for(layer_name in names(layer_contributions)){
  coeff<-layer_contributions[[layer_name]]
  activation<-get_layer(model,layer_name)$output
  scaling<-k_prod(k_cast(k_shape(activation),'float32'))
  loss<-loss+(coeff*k_sum(k_square(activation))/scaling)
}
dream<-model$input
grads<-k_gradients(loss,dream)[[1]]
grads<-grads/k_maximum(k_mean(k_abs(grads)),1e-7)
outputs<-list(loss,grads)
fetch_loss_and_grads<-k_function(list(dream),outputs)
eval_loss_and_grads<-function(x){
  outs<-fetch_loss_and_grads(list(x))
  loss_value<-outs[[1]]
  grad_values<-outs[[2]]
  list(loss_value,grad_values)
}
gradient_ascent<-function(x,iterations,step,max_loss=NULL){
  for(i in 1:iterations){
    c(loss_value,grad_values)%<-%eval_loss_and_grads(x)
    if(!is.null(max_loss)&&loss_value>max_loss)
      break
    cat('..loss value at',i,':',loss_value,'\n')
    x<-x+(step*grad_values)
  }
  x
}
step<-0.01
num_octave<-3
octave_scale<-1.4
iterations<-20
max_loss<-10
base_image_path<-"C:/Users/USER/Desktop/ZFAImt_k.jpeg"
img<-preprocess_image(base_image_path)
original_shape<-dim(img)[-1]
successive_shapes<-list(original_shape)
for(i in 1:num_octave){
  shape<-as.integer(original_shape/(octave_scale^i))
  successive_shapes[[length(successive_shapes)+1]]<-shape
}
successive_shapes<-rev(successive_shapes)
original_img<-img
shrunk_original_img<-resize_img(img,successive_shapes[[1]])
for(shape in successive_shapes){
  cat('processing image shape',shape,'\n')
  img<-resize_img(img,shape)
  img<-gradient_ascent(img,iterations=iterations,step=step,max_loss = max_loss)
  upscaled_shrunk_original_img<-resize_img(shrunk_original_img,shape)
  same_size_original<-resize_img(original_img,shape)
  lost_detail<-same_size_original-upscaled_shrunk_original_img
  img<-img+lost_detail
  shrunk_original_img<-resize_img(original_img,shape)
  save_image(img,fname=sprintf('dream_at_scale_%s.png',paste(shape,collapse = 'x')))
}
resize_img<-function(img,size){
  image_array_resize(img,size[[1]],size[[2]])
}
save_image<-function(img,fname){
  img<-deprocess_image(img)
  image_array_save(img,fname)
}
preprocess_image<-function(image_path){
  image_load(image_path)%>%image_to_array()%>%array_reshape(dim=c(1,dim(.)))%>%inception_v3_preprocess_input()
}
deprocess_image<-function(img){
  img<-array_reshape(img,dim = c(dim(img)[[2]],dim(img)[[3]],3))/2
  img<-img+0.5
  img<-img*255
  dims<-dim(img)
  img<-pmax(0,pmin(img,255))
  dim(img)<-dims
  img
}

