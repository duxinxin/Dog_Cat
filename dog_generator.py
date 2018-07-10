import keras
import cv2
import threading
import numpy as np

class CDGenerator(keras.utils.Sequence):
	def __init__(self,data,n,name,des_size=(224,224),means=None,stds=None,is_directory=True,batch_size=32,shuffle=True,seed=0):
		self.x=np.array(data[0])
		self.name = name
		if len(data)>=2:
			self.y = np.array(data[1])
		else:
			self.y = None
# 			if self.name == 'test':
# 				print("first",self.y)
		self.n=n
		self.des_size=des_size
		self.is_directory=is_directory
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.lock=threading.Lock()
		self.index_array=self._set_index_array()
		self.means=means
		self.stds=stds

	def reset_index(self):
		self.batch_index=0

	def _set_index_array(self):
		self.index_array=np.arange(self.n)
		if self.shuffle:
			np.random.shuffle(self.index_array)

	def on_epoch_end(self):
		self._set_index_array()

	def __len__(self):
		return int(np.ceil(self.n/self.batch_size))

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
			
