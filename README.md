### 数据集
数据下载地址：[数据集](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

### 读取数据
因为图片中有数据的类别，所以直接读取图片。例：dog.1000.jpg,cat.1.jpg.


```
def load_data(path):
    img_names=[]
    img_labels=[]
    for dpath,dnames,fnames in os.walk(path):
#         img_names=fnames
#         print(fnames)
        for i in fnames:
            img_names.append("train/" + i)
            img_labels.append(i[:3])
#     print(img_names[-20:])
#     print(img_labels[-20:])
    train_x,valid_y,labels_x,labels_y = train_test_split(img_names,img_labels,test_size=0.3,random_state=42)

#     cate_dict={"cat":0,"dog":1}
#     print(labels_x[:30])
#     print(labels_y[:30])
#     print(len(labels_x))
    train_labels=[]
    valid_labels=[]
    for i in labels_x:
        if i == 'cat':
            train_labels.append(0)
        else:
            train_labels.append(1)
       
    for j in labels_y:
        if j == 'cat':
            valid_labels.append(0)
        else:
            valid_labels.append(1)
#     print(len(train_labels),len(valid_labels))
#     print(valid_labels[:30])

    return train_x,valid_y,train_labels,valid_labels
```
### 自定义生成器
在这里我是根据keras源码中的生成器自定义了一个生成器，直接继承了keras.utils.Sequence类，核心是在__getitem__中做了实现。

```
def __getitem__(self,idx):
		if self.index_array is None:
			self._set_index_array()
		index_array = self.index_array[self.batch_size*idx:
										self.batch_size*(idx+1)]
		return self._data_generate(index_array)

	def _data_generate(self,index_array):
		imgs = np.zeros((len(index_array),self.des_size[0],self.des_size[1],3),dtype=np.uint8)
		labels = None
		if self.is_directory:
			img_names = self.x[index_array]
			for name_index in range(len(img_names)):
				img = cv2.imread(img_names[name_index])
				if img is not None:
					img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					img = cv2.resize(img,self.des_size)
# 					print(img)                    
					imgs[name_index]=img
		else:
			for i in range(len(index_array)):
				img = self.x[index_array[i]]
				img = cv2.resize(img,self.des_size)
				imgs[i] = img

# 		if self.name =='test':
# 			print("final:",self.y)
		if self.y is not None:
			if self.name =='test':
				pass
# 				print('test',self.y)
			else:
				labels=self.y[index_array]
# 		labels = keras.utils.to_categorical(labels, num_classes=2)
# 		print(img_names,labels)
		if labels is None:
			return imgs
		else:
			return imgs,labels
```
__len__是总共有多少个batch

_set_index_array 重置 index_array 保证每一个epoch都取到不同位置的batch，随机性更大，模型的泛化性也会更大

### 迁移学习
使模型在imagenet上进行训练，然后修改输出层，在自己的数据集上再次进行训练。

```
Input = keras.layers.Input(shape=(224,224,3))
process_input = keras.layers.Lambda(res_preprocess_input)(Input)
resnet = resnet50.ResNet50(include_top=False,weights="imagenet",input_tensor=process_input,pooling="avg")
resModel = Model(inputs=resnet.input,outputs=resnet.output)
output = resModel.output
output=Dropout(0.5)(output)
predictions = Dense(1,activation="sigmoid")(output)
model = Model(inputs=resModel.input, outputs=predictions)

for layers in resnet.layers:
    layers.trainable=False
    
model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator,len(train_img)//batch_size,
                    epochs=epochs,verbose=1,
                    validation_data=valid_generator,
                    validation_steps=len(valid_img)// batch_size)
```
### 特征向量的导出
导出每个网络微调后的特征向量，全局平均池化层后的输出。
(n,2048)

```
import h5py

def output_features(model_name,model,train_generator,train_labels,valid_generator,valid_labels,test_generator,batch_size=32):
    if model_name == 'resnet_features':
        model.load_weights('resnet_weights.h5',by_name=True)
    elif model_name == 'ince_v3_features':
        model.load_weights('ince_v3_weights.h5',by_name=True)
    else:
        model.load_weights('xception_weights.h5',by_name=True)
    
    #将labels转成array
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    
    #trian、valid、test的features
    train_features = model.predict_generator(train_generator,int(np.ceil(train_generator.n/batch_size)),verbose=1)
    valid_features = model.predict_generator(valid_generator,int(np.ceil(valid_generator.n/batch_size)),verbose=1)
    test_features = model.predict_generator(test_generator,int(np.ceil(test_generator.n/batch_size)),verbose=1)
    
    with h5py.File(model_name+'.h5','w') as file:
        file.create_dataset('train',data=train_features,dtype='float32')
        file.create_dataset('train_labels',data=np.array(train_generator.y),dtype='uint8')
        file.create_dataset('valid',data=valid_features,dtype='float32')
        file.create_dataset('valid_labels',data=np.array(valid_generator.y),dtype='uint8')
        file.create_dataset('test',data=test_features,dtype='float32')
```

```
# inception
Input = keras.layers.Input(shape=(299,299,3))
process_input = keras.layers.Lambda(incep_preprocess_input)(Input)
ince_v3 = inception_v3.InceptionV3(include_top=False,weights="imagenet",input_tensor=process_input,pooling="avg")
print(ince_v3.output)
output_features('ince_v3_features',Model(inputs=ince_v3.input,outputs=ince_v3.output),
                train_generator,train_labels,valid_generator,valid_labels,test_generator)
```
### 模型的融合
结合上面导出的特征，在融合的特征上训练一个分类的模型
```
feature_files = ['resnet_features.h5','ince_v3_features.h5','xception_features.h5']

X_train =[]
y_train =[]
X_valid =[]
y_valid =[]
X_test =[]

for file_name in feature_files:
    with h5py.File(file_name,'r') as h:
        X_train.append(np.array(h['train']))
        y_train = (np.array(h['train_labels']))
        
        X_test.append(np.array(h['test']))
        
        X_valid.append(np.array(h['valid']))
        y_valid = (np.array(h['valid_labels']))
        print(np.array(h['train']).shape,np.array(h['valid']).shape,np.array(h['test']).shape)
#         print(X_train[:3],y_train[:3])
train = np.concatenate(X_train,axis=1)
valid = np.concatenate(X_valid,axis=1)
test  = np.concatenate(X_test, axis=1)
print(train.shape,valid.shape,test.shape)
```
### 训练
训练融合后的新模型

```
import keras.utils

input_tensor = Input(X_train.shape[1:])
print(input_tensor)
# print(X_train)
x = Dropout(0.5)(input_tensor)
output = Dense(1,activation='sigmoid')(x)

con_model = Model(inputs=input_tensor,outputs=output)
con_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

con_model.fit(X_train,y_train,batch_size=128, epochs=5,validation_data=(valid,y_valid))
```

### 总结
当使用单个模型时的精确率很难再会有提高时，可以尝试将模型融合，也许会得到更好的效果。