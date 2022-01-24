import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
from scipy.ndimage.interpolation import shift
import os

train_images = np.load("a-z_train_images.npz")['arr_0']
train_labels = np.load("a-z_train_labels.npz")['arr_0']

val_images = np.load("a-z_test_images.npz")['arr_0']
val_labels = np.load("a-z_test_labels.npz")['arr_0']

print(train_images.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_labels.shape)

if K.image_data_format() == "channels_first":
    shape = (1,28,28)

    train_images = train_images.reshape(train_images.shape[0],1,28,28)
    val_images = val_images.reshape(val_images.shape[0],1,28,28)
    
    
else:
    shape = (28,28,1)
    train_images = train_images.reshape(train_images.shape[0],28,28,1)
    val_images = val_images.reshape(val_images.shape[0],28,28,1)
    
train_images = train_images/255.0
val_images = val_images/255.0

print(train_images.shape)


import random
import matplotlib.pyplot as plt

idx = random.randint(0, len(train_images))
plt.imshow(train_images[idx, :])
plt.show()


import random
import matplotlib.pyplot as plt

idx = random.randint(0, len(val_images))
plt.imshow(val_images[idx, :])
plt.show()


data = ImageDataGenerator(rotation_range=5,zoom_range=0.2)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3),padding="Same", input_shape = (28,28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(64, (3, 3),padding="Same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3),padding="Same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 26, activation = 'softmax'))


classifier.compile(optimizer ="adam", loss = "sparse_categorical_crossentropy",metrics=['accuracy'])

print(classifier.summary())


callbacks = [
    keras.callbacks.EarlyStopping(patience=2),
    keras.callbacks.ModelCheckpoint(filepath='classifier.{epoch:02d}-{val_loss:.2f}.h5'),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]



history = classifier.fit(data.flow(train_images, train_labels,shuffle = True,batch_size=32), epochs=15, 
                    validation_data=(val_images, val_labels), callbacks = callbacks)



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

val_loss, val_acc = classifier.evaluate(val_images,  val_labels, verbose=2)


fname = "a-z_cnn_model.h5"
classifier.save(fname)



loaded_model = keras.models.load_model("a-z_cnn_model.h5")
loaded_model.build()
loaded_model.summary()


test_images = np.load("test_alpha.npz")['arr_0']


path = r'E:\RawatTech\A2Z_dataset'
catagories = os.listdir(path)



if K.image_data_format() == "channels_first":
    shape = (1,28,28)
    test_images = test_images.reshape(test_images.shape[0],1,28,28)
    
else:
    shape = (28,28,1)
    test_images = test_images.reshape(test_images.shape[0],28,28,1)
   
    
test_images = test_images/255.0

print(test_images.shape)



pred = loaded_model.predict(test_images)
pred[:5]


classes = [np.argmax(element) for element in pred]
print(classes)

confi = []
for i in pred:
    maximum = max(i)
    confi.append(maximum)


catagory = []
for x in classes:
    catagory.append(catagories[x])
    
print(catagory)
    
   
import pandas as pd    

df = pd.DataFrame({'classes':classes,'alphabets':catagory,'confidence':confi})


df.to_excel("alpha_result.xlsx",index = False)























