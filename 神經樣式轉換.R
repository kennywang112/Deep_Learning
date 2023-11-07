library(keras)
library(tidyverse)
library(tensorflow)
tf$compat$v1$disable_eager_execution()
target_image_path<-"C:/Users/USER/Desktop/S__10797109.jpg"
style_reference_image_path<-"C:/Users/USER/Desktop/000.jpg"
img<-image_load(target_image_path)
width<-img$size[[1]]
height<-img$size[[2]]
img_nrows<-400
img_ncols<-as.integer(width*img_nrows/height)
preprocess_image<-function(path){
  img<-image_load(path,target_size = c(img_nrows,img_ncols))%>%
    image_to_array()%>%
    array_reshape(c(1,dim(.)))
  imagenet_preprocess_input(img)
}
deprocess_image<-function(x){
  x<-x[1,,,]
  x[,,1]<-x[,,1]+103.939
  x[,,2]<-x[,,2]+116.779
  x[,,3]<-x[,,3]+123.68
  x<-x[,,c(3,2,1)]
  x[x>255]<-255
  x[x<0]<-0
  x[]<-as.integer(x)/255
  x
}
target_image<-k_constant(preprocess_image(target_image_path))
style_reference_image<-k_constant(
  preprocess_image(style_reference_image_path)
)
combination_image<-k_placeholder(c(1,img_nrows,img_ncols,3))
input_tensor<-k_concatenate(list(target_image,style_reference_image,combination_image),axis = 1)
model<-application_vgg19(input_tensor = input_tensor,weights = 'imagenet',include_top = FALSE)
cat('model loaded\n')
content_loss<-function(base,combination){
  k_sum(k_square(combination-base))
}
gram_matrix<-function(x){
  features<-k_batch_flatten(k_permute_dimensions(x,c(3,1,2)))
  gram<-k_dot(features,k_transpose(features))
  gram
}
style_loss<-function(style,combination){
  S<-gram_matrix(style)
  C<-gram_matrix(combination)
  channel<-3
  size<-img_nrows*img_ncols
  k_sum(k_square(S-C))/(4*channel^2*size^2)
}
total_variation_loss<-function(x){
  y_ij<-x[,1:(img_nrows-1L),1:(img_ncols-1L),]
  y_i1j<-x[,2:(img_nrows),1:(img_ncols-1L),]
  y_ij1<-x[,1:(img_nrows-1L),2:(img_ncols),]
  a<-k_square(y_ij-y_i1j)
  b<-k_square(y_ij-y_ij1)
  k_sum(k_pow(a+b,1.25))
}
outputs_dict<-lapply(model$layers,`[[`,'output')
names(outputs_dict)<-lapply(model$layers,`[[`,'name')
content_layer<-20
style_layers=c(3,6,9,14,19)
total_variation_weight<-1e-4
style_weight<-1.0
content_weight<-0.025
loss<-k_variable(0.0)
layer_features<-outputs_dict[[content_layer]]
target_image_features<-layer_features[1,,,]
combination_features<-layer_features[3,,,]
loss<-loss+content_weight*content_loss(target_image_features,combination_features)
for(layer_name in style_layers){
  layer_features<-outputs_dict[[layer_name]]
  style_reference_features<-layer_features[2,,,]
  combination_features<-layer_features[3,,,]
  sl<-style_loss(style_reference_features,combination_features)
  loss<-loss+((style_weight/length(style_layers))*sl)
}
loss<-loss+(total_variation_weight*total_variation_loss(combination_image))

grads<-k_gradients(loss,combination_image)[[1]]
fetch_loss_and_grads<-k_function(list(combination_image),list(loss,grads))
eval_loss_and_grads<-function(image){
  image<-array_reshape(image,c(1,img_nrows,img_ncols,3))
  outs<-fetch_loss_and_grads(list(image))
  list(
    loss_value=outs[[1]],
    grad_values=array_reshape(outs[[2]],dim=length(outs[[2]]))
    )
}
library(R6)
Evaluator<-R6Class('Evaluator',
  public=list(
    loss_value=NULL,
    grad_values=NULL,
    initialize=function(){
      self$loss_value<-NULL
      self$grad_values<-NULL
      },
  loss=function(x){
    loss_and_grad<-eval_loss_and_grads(x)
    self$loss_value<-loss_and_grad$loss_value
    self$grad_values<-loss_and_grad$grad_values
    self$loss_value
    },
  grads=function(x){
    grad_values<-self$grad_values
    self$loss_value<-NULL
    self$grad_values<-NULL
    grad_values
    }
  )
)
evaluator<-Evaluator$new()

iterations<-20
dms<-c(1,img_nrows,img_ncols,3)
x<-preprocess_image(target_image_path)
x<-array_reshape(x,dim = length(x))
for(i in 1:iterations){
  opt<-optim(
    array_reshape(x,dim=length(x)),
    fn=evaluator$loss,
    gr=evaluator$grads,
    method = 'L-BFGS-B',
    control = list(maxit=15)
    )
  cat('loss:',opt$value,'\n')
  image<-x<-opt$par
  image<-array_reshape(image,dms)
  im<-deprocess_image(image)
  plot(as.raster(im))
}









