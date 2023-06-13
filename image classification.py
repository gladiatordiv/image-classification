#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import Sequential,datasets
from tensorflow.keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,MaxPool2D,BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()


# In[3]:


X_train.shape


# In[4]:


y_train=y_train.reshape(-1,)


# In[5]:


y_train


# In[6]:


y_classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[7]:


len(y_classes)


# In[8]:


def showimage(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(y_classes[y[index]])


# In[9]:


showimage(X_train,y_train,5)


# In[10]:


X_train=X_train/255
X_test=X_test/255


# In[11]:


X_train


# In[12]:


model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(4,4),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=34,activation="relu"))
model.add(Dense(units=10,activation="softmax"))


# In[13]:


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[14]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5)


# In[15]:


model5 = Sequential()
model5.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model5.add(BatchNormalization())
model5.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model5.add(BatchNormalization())
model5.add(MaxPool2D((2, 2)))
model5.add(Dropout(0.2))
model5.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model5.add(BatchNormalization())
model5.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model5.add(BatchNormalization())
model5.add(MaxPool2D((2, 2)))
model5.add(Dropout(0.3))
model5.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model5.add(BatchNormalization())
model5.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model5.add(BatchNormalization())
model5.add(MaxPool2D((2, 2)))
model5.add(Dropout(0.4))
model5.add(Flatten())
model5.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model5.add(BatchNormalization())
model5.add(Dropout(0.5))
model5.add(Dense(10, activation='softmax'))
model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history5=model5.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))


# In[43]:


y_pred=model5.predict(X_test)
y_pred[9]


# In[44]:


y_pred=[np.argmax(arr) for arr in y_pred]
y_pred


# In[49]:


y_tes=y_test.reshape(-1, )
y_pred[3]


# In[50]:


showimage(X_test,y_tes,0)


# In[51]:


model5.evaluate(X_test,y_test)


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




