# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:45:15 2020

@author: ASUS
"""

import tensorflow as tf
import os

directory="brain_tumor_dataset/"
print(len(os.listdir("brain_tumor_dataset/yes")))
print(len(os.listdir("brain_tumor_dataset/no")))

os.mkdir("training/")
os.mkdir("training/yes")
os.mkdir("training/no")
os.mkdir("testing/")
os.mkdir("testing/yes")
os.mkdir("testing/no")

import train_test_set as tst
ydir="brain_tumor_dataset/yes/"
ndir="brain_tumor_dataset/no/"

ytrain="training/yes/"
ntrain="training/no/"

ytest="testing/yes/"
ntest="testing/no/"

tst.split(ydir,ytrain,ytest,0.9)
tst.split(ndir,ntrain,ntest,0.9)

print(len(os.listdir(ytrain)))
print(len(os.listdir(ydir)))



train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.4,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
       
        )
testdatagen=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.4,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        
        )

train_dataset=train_datagen.flow_from_directory(directory="training/",
                                                target_size=(224,224),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                batch_size=32
                                                )
test_set=testdatagen.flow_from_directory(directory="testing/",
                                         target_size=(224,224),
                                         class_mode='binary',
                                         color_mode='rgb',
                                         batch_size=16
                                         
                                         )


vggmodel=tf.keras.applications.vgg16.VGG16(include_top=False,
                                           weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                           input_shape=(224,224,3)
                                           )

for layer in vggmodel.layers:
    layer.trainable=False

vggmodel.summary()

model=tf.keras.Sequential([
        vggmodel,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1,activation='sigmoid')
        ])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=['accuracy'],loss='binary_crossentropy')


history=model.fit_generator(train_dataset,steps_per_epoch=50,epochs=120,validation_data=test_set,
                            validation_steps=30,
                            verbose=1
                            )

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r','Training Accuracy')
plt.plot(epochs,val_acc,'b','Training Accuracy')
plt.title("Training and validation Accuracy")
plt.figure()

plt.plot(epochs,loss,'r','Training Accuracy')
plt.plot(epochs,val_loss,'b','Training Accuracy')
plt.title("Training and validation Accuracy")
plt.figure()

model.save("testmodel1.h5")

