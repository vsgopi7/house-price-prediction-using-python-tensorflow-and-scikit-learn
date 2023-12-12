#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
df = pd.read_csv('house_data.csv') 
df.head()


# In[2]:


df.tail()


# In[3]:


df.shape


# In[4]:


df.hist("price")
plt.show()


# In[5]:


df.isna()


# In[6]:


df.isna().sum()


# In[7]:


df = df.iloc[:,1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()


# In[8]:


X = df_norm.iloc[:, :5]
X.head()


# In[9]:


Y = df_norm.iloc[:, -1]
Y.head()


# In[10]:


X_arr = X.values
Y_arr = Y.values
X_arr


# In[11]:


Y_arr


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.01, shuffle = True, random_state=1)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[13]:


def get_model():
    
    model = Sequential([
        Dense(10, input_shape = (5,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    
    return model


# In[14]:


model = get_model()
model.summary()


# In[21]:


model = get_model()

# this prediction is before training the model
preds_on_untrained = model.predict(X_test)
# Train model and store in the object history
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000
)


# In[20]:


# plot history
plot_loss(history)
# make predictions on the trained model
preds_on_trained = model.predict(X_test)
compare_predictions(preds_on_untrained, preds_on_trained, y_test)
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_actual(pred):
    return int(pred * y_std + y_mean)

print(convert_label_actual(-1.836486))
price_on_untrained = [convert_label_actual(y) for y in preds_on_untrained]
price_on_trained = [convert_label_actual(y) for y in preds_on_trained]
price_y_test = [convert_label_actual(y) for y in y_test]
# plot price predictions
compare_predictions(price_on_untrained, price_on_trained, price_y_test)


# In[ ]:





# In[ ]:




