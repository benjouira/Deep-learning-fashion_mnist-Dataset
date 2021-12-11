import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score


print(tf.__version__)


# ********************************************************************
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images_f, train_labels_f), (test_images, test_labels) = fashion_mnist.load_data()

# ********************************************************************
print (train_images_f.shape, train_labels_f.shape, test_images.shape, test_labels.shape)


# ********************************************************************
train_images_f = train_images_f / 255
test_images = test_images / 255
train_images_f[1,12:17,10:13]


# ********************************************************************
train_images, val_images, train_labels, val_labels = train_test_split(train_images_f , train_labels_f , test_size=0.1, random_state=1)
print(train_images.shape , val_images.shape)

# ********************************************************************
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(5,5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel ( class_names [train_labels[i]])

# ********************************************************************
cls_fashion = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(128,activation= tf.nn.relu),
tf.keras.layers.Dense(128,activation= tf.nn.relu),
tf.keras.layers.Dense(128,activation= tf.nn.relu),
tf.keras.layers.Dense(128,activation= tf.nn.relu),
tf.keras.layers.Dense(128,activation= tf.nn.relu),
tf.keras.layers.Dense(10,activation= tf.nn.softmax),
])
cls_fashion.summary()



# ********************************************************************
cls_fashion.compile(optimizer='adam', loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# ********************************************************************
history = cls_fashion.fit (train_images,train_labels , epochs= 10 , validation_data= (val_images,val_labels))
# by default batch_size = non same as batch_size=32

# ***************************************

# Validation du modèle
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,2) # set the vertical range to [0,1]
plt.show()

# ***************************************

pred_labels= cls_fashion.predict(test_images)
print(pred_labels.shape)
pred_labels= np.argmax(pred_labels ,axis=1)
# a7na 3ndna 10 class besh nchoufou el class es7i7 howa eli fih valeur max
# argmax = l'indice de maximam

cm = confusion_matrix(test_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
