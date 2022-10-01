import shutil

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import TensorBoard
import time
import os
from sklearn.model_selection import KFold

#tensorboard --logdir="logs/"

def Convert_Categorical_to_Numeric(PandasSeries,interpolate=False):

    if (interpolate):
        #assert PandasSeries.isnull().values.any()==True,"Given column posses no nan values"
        PandasSeries.interpolate(inplace=True)
        PandasSeries.replace(PandasSeries.unique(),[i for i in range(PandasSeries.unique().shape[0])],inplace=True)
    else:
        assert PandasSeries.isnull().values.any()==False,"Given column posses some nan values, use INTERPOLATE flag"
        PandasSeries.replace(PandasSeries.unique(), [i for i in range(PandasSeries.unique().shape[0])], inplace=True)

    return PandasSeries



with tf.device ("/GPU:0"):




    train_df =pd.read_csv(r"D:\System_folders\Dowloads\titanic\train.csv")
    #train_df=pd.read_csv(r"C:\Users\Uzer\Downloads\train.csv")
    #test_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\test.csv")
    #gend_sub_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\gender_submission.csv")

    train_df.drop(["Name","PassengerId","Ticket"],axis=1,inplace=True)

    #print(train_df.head())
    train_df["Cabin"]=Convert_Categorical_to_Numeric(train_df["Cabin"],interpolate=True)
    train_df["Embarked"]=Convert_Categorical_to_Numeric(train_df["Embarked"],interpolate=True)
    train_df["Sex"]=Convert_Categorical_to_Numeric(train_df["Sex"])
    train_df.interpolate(inplace=True)

    #print(train_df.isnull().sum())

    X=train_df.drop("Survived",axis=1).to_numpy()
    Y=train_df["Survived"].to_numpy()

    def create_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),#показала лучший результат
        ])


    shutil.rmtree("logs\\fit")
    shutil.rmtree("models\\")
    os.mkdir("models")
    os.mkdir("logs\\fit")

    model=create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),#normal rate =0.001
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    n_split=10
    epochs=80

    time_stop=int(time.time())
    tensorboard=TensorBoard(log_dir="logs\\fit\\{}".format(time_stop))
    for train_index,valid_index in KFold(n_split).split(X):

        x_train,x_valid=X[train_index],X[valid_index]
        y_train,y_valid=Y[train_index],Y[valid_index]

        model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            validation_data=[x_valid,y_valid],
            #batch_size=32,
            callbacks=[tensorboard]
        )
    model.save("models\\{}".format(time_stop))






