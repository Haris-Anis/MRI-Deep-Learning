import tensorflow as tf
import h5py
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_dataset=[]
y_dataset=[]
for i in range(len(os.listdir("Brain Tumor type dataset"))):
    file=h5py.File("Brain Tumor type dataset/"+str(i+1)+".mat",mode='r')
    Dataset=file["cjdata/image"]
    Label=file["cjdata/label"]
    x=np.array(Dataset,dtype='float32')
    y=np.array(Label,dtype='float32')
    x=scaler.fit_transform(x)
    x=np.repeat(x[:,:,np.newaxis],3,axis=2)
    x=cv2.resize(x,(224,224))
    x_dataset.append(x)
    y_dataset.append(y)

import matplotlib.pyplot as plt
plt.imshow(X_dataset[58])
X_dataset=np.array(x_dataset)
Y_dataset=np.array(y_dataset)
Y_dataset=Y_dataset.reshape((3064,1))

Y_dataset=Y_dataset-1

from sklearn.model_selection import  StratifiedKFold
folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(X_dataset,Y_dataset))
Inception=tf.keras.applications.InceptionV3(include_top=False,input_shape=(224,224,3))
input_image=tf.keras.layers.Input((224,224,3))
x=Inception (input_image)
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(3)(x)
out=tf.keras.layers.Activation(activation='softmax')(x)

model=tf.keras.Model(inputs=input_image,outputs=out)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0002),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
datagen=tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.4,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2
       
        )
for j,(train_idx,val_idx) in enumerate(folds):
    print("Fold "+str(j+1))
    
    x_train=X_dataset[train_idx]
    y_train=Y_dataset[train_idx]
    x_val=X_dataset[val_idx]
    y_val=Y_dataset[val_idx]
    generator=datagen.flow(x_train,y_train,batch_size=30)
    model.fit_generator(generator,steps_per_epoch=32,epochs=10,validation_data=(x_val,y_val))
    

model.save_weights("accu95brain.h5")



"""
model=tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(64,64)),
          tf.keras.layers.Activation(activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3)),
          tf.keras.layers.Activation(activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3)),
          tf.keras.layers.Activation(activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64,activation='relu'),
          tf.keras.layers.Dense(3,activation='softmax')
        ])
    
summar=model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


history=model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test))