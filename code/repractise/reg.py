import numpy as np
import pandas as pd

df = pd.read_csv("electric_vehicle.csv")
df.sample(2)

df.size

df.drop("VIN (1-10)",axis="columns",inplace=True)

df.sample(1)

df.columns

df.County.unique()

df = pd.read_csv("train.csv")

df.sample(2)

df.describe()

df.info()

df.Month.unique()

df_encoded = pd.get_dummies(df, columns = ['Month'])

df_encoded.sample(2)

df.City.unique()

df_encoded.drop("City",axis='columns',inplace=True)

df_encoded.sample(2)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

df_encoded.shape[1]

x = df_encoded.drop("Healthcare_Index",axis="columns")
y = df_encoded["Healthcare_Index"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x.sample(2)

class_col = x.columns
for i in range(0,len(class_col)):
  print(df_encoded[class_col[i]].unique())

class_col = x.columns
df_encoded[class_col[2]].unique()

df_encoded = pd.get_dummies(df_encoded, columns = ['Traffic_Density'])

x = df_encoded.drop("Healthcare_Index",axis="columns")
y = df_encoded["Healthcare_Index"]

x

x = scaler.fit_transform(x)   # Column Month and Traffic_Density are one hote encoded
# Right now does not transform y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x[0]

import tensorflow as tf
from tensorflow import keras

x.shape[1]

my_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

model = keras.Sequential([
    keras.layers.Dense(22,input_shape=(x.shape[1],),activation="relu"),
    keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
),
    keras.layers.Dense(24,activation="relu"),
    keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
),
    keras.layers.Dense(20,activation="linear"),
    keras.layers.Dense(1,activation="linear")
])
model.compile(
    optimizer="adam",
    loss = "mse",
    metrics = [keras.metrics.RootMeanSquaredError()]
)
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=my_callback,epochs=1000)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(history.history)

y_pred = model.predict(x_test)

model.evaluate(x_test,y_test)

y_pred[:5]

y_test[:5]

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

y_test.describe()

y_train.describe()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

df1 = pd.read_csv("test.csv")

df1.sample(2)

df1_encoded = pd.get_dummies(df1, columns = ['Month'])

df1_encoded.drop("City",axis='columns',inplace=True)

df1_encoded.columns

df_encoded.columns

x = df1_encoded.drop("Healthcare_Index",axis="columns")
y = df1_encoded["Healthcare_Index"]

x.columns

df1_encoded.sample(2)

df1_encoded = pd.get_dummies(df1_encoded, columns = ['Traffic_Density'])

df1_encoded.columns

df_encoded.columns

x1 = df1_encoded.drop("Healthcare_Index",axis="columns")
y1 = df1_encoded["Healthcare_Index"]

x1 = scaler.fit_transform(x1)

x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2,random_state=42)

model = keras.Sequential([
    keras.layers.Dense(22,input_shape=(x1.shape[1],),activation="relu"),
    keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
),
    keras.layers.Dense(24,activation="relu"),
    keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
),
    keras.layers.Dense(20,activation="linear"),
    keras.layers.Dense(1,activation="linear")
])
model.compile(
    optimizer="adam",
    loss = "mse",
    metrics = [keras.metrics.RootMeanSquaredError()]
)
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=my_callback,epochs=1000)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_pred = model.predict(x_test)

model.evaluate(x_test,y_test)

y_pred[:5]

y_test[:5]

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

feature_list = ['Month_October','Month_November','Month_December']

help = pd.DataFrame(0, index=np.arange(867), columns=feature_list)

help.sample(2)

df_concat = pd.concat([df1_encoded, help], axis=1)

df_concat.shape[1]

df_concat.columns

df1_encoded.shape

df1_encoded = pd.get_dummies(df_concat, columns = ['Traffic_Density'])

df1_encoded = df1_encoded.dropna()

df1_encoded.shape[1]

x1 = df1_encoded.drop("Healthcare_Index",axis="columns")
y1 = df1_encoded["Healthcare_Index"]

x1 = scaler.fit_transform(x1)

y_pred = model.predict(x1)

model.evaluate(x1,y1)

y_pred[:5]

y1[:5]

r2_score(y1,y_pred)

y1.info()

y1.iloc[8]