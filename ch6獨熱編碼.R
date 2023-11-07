#字符級別獨熱編碼
samples<-c('the cat sat on the mat.','the dog ate my homework.')
token_index<-list()
for(sample in samples){
  for(word in strsplit(sample,' ')[[1]]){
    if(!word %in% names(token_index)){
      token_index[[word]]<-length(token_index)+2}
  }
}
max_length<-10
results<-array(0,dim=c(length(samples),max_length,max(as.integer(token_index))))#dim(2,10,10)
for(i in 1:length(samples)){
  sample<-samples[[i]]
  words<-head(strsplit(sample,' ')[[1]],n=max_length)
  for(j in 1:length(words)){
    index<-token_index[[words[[j]]]]
    results[[i,j,index]]<-1
  }
}
#字符級獨熱編碼
ascii_tokens<-c('',sapply(as.raw(c(32:126)),rawToChar))
token_index<-c(1:(length(ascii_tokens)))
names(token_index)<-ascii_tokens
max_length<-50
results<-array(0,dim=c(length(samples),max_length,length(token_index)))
for(i in 1:length(samples)){
  sample<-samples[[i]]
  characters<-strsplit(sample,'')[[1]]
  for(j in 1:length(characters)){
    character<-characters[[j]]
    results[i,j,token_index[[character]]]<-1
  }
}
#keras內置單詞級獨熱編碼
library(keras)
tokenizer<-text_tokenizer(num_words = 1000)%>%fit_text_tokenizer(samples)#創建最常見的1000單詞、生成單字索引
sequences<-texts_to_sequences(tokenizer,samples)#字符轉變為整數索引列表
one_hot_results<-texts_to_matrix(tokenizer,samples,mode = 'binary')#直接獲取讀熱二進制
word_index<-tokenizer$word_index#恢復以計算的單詞索引
cat('found',length(word_index),'unique tokens.\n')

#處理原始數據，imdb單詞嵌入
#數據收集到標籤列表中
train_dir<-file.path('C:/Users/USER/Desktop/Deep Learning/imdb/train')
labels<-c()
texts<-c()
for(label_type in c('neg','pos')){
  label<-switch(label_type,neg=0,pos=1)
  dir_name<-file.path(train_dir,label_type)
  for(fname in list.files(dir_name,pattern = glob2rx('*.txt'),full.names=TRUE)){
    texts<-c(texts,readChar(fname,file.info(fname)$size))
    labels<-c(labels,label)
  }
}
#標籤化數據
maxlen<-100#100以後不考慮
training_samples<-200#200個樣本上訓練
validation_samples<-10000#10000個樣本上驗證
max_words<-10000#考慮前10000個單字
tokenizer<-text_tokenizer(num_words = max_words)%>%fit_text_tokenizer(texts)
sequences<-texts_to_sequences(tokenizer,texts)
word_index=tokenizer$word_index
cat(length(word_index))
data<-pad_sequences(sequences,maxlen = maxlen)
labels<-as.array(labels)
cat(dim(data))
cat(dim(labels))
indices<-sample(1:nrow(data))
training_indices<-indices[1:training_samples]
validation_indices<-indices[(training_samples+1):(training_samples+validation_samples)]
x_train<-data[training_indices,]
y_train<-labels[training_indices]
x_val<-data[validation_indices,]
y_val<-labels[validation_indices]
#預處理遷入
glove_dir<-"C:/Users/USER/Desktop/Deep Learning/glove.6b"
lines<-readLines(file.path(glove_dir,'glove.6B.100d.txt'))
embeddings_index<-new.env(hash = TRUE,parent = emptyenv())
for(i in 1:2000){
  line<-lines[[i]]
  values<-strsplit(line," ")[[1]]
  word<-values[[1]]
  embeddings_index[[word]]<-as.double(values[-1])
}
cat(length(embeddings_index))
embedding_dim<-100
embedding_matrix<-array(0,c(max_words,embedding_dim))
for(word in names(word_index)){
  index<-word_index[[word]]
  if(index<max_words){
    embedding_vector<-embeddings_index[[word]]
    if(!is.null(embedding_vector))
      embedding_matrix[index+1,]<-embedding_vector
  }
}
model<-keras_model_sequential()%>%
  layer_embedding(input_dim = max_words,output_dim = embedding_dim,input_length = maxlen)%>%
  layer_flatten()%>%
  layer_dense(units = 32,activation = 'relu')%>%
  layer_dense(units=1,activation = 'sigmoid')
summary(model)
get_layer(model,index=1)%>%set_weights(list(embedding_matrix))%>%freeze_weights()
model%>%compile(
  optimizer='rmsprop',loss='binary_crossentropy',metrics=c('acc')
)
history<-model%>%fit(x_train,y_train,epoch=20,batch_size=32,validation_data=list(x_val,y_val))
