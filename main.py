import shutil

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import TensorBoard
import time




with tf.device ("/GPU:0"):




    train_df =pd.read_csv(r"D:\System_folders\Dowloads\titanic\train.csv")
    #train_df=pd.read_csv(r"C:\Users\Uzer\Downloads\train.csv")
    #test_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\test.csv")
    #gend_sub_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\gender_submission.csv")

    train_df.drop(["Name","Ticket","Cabin","Embarked","PassengerId","Age","SibSp","Parch"],axis=1,inplace=True)

    train_df.loc[train_df["Sex"]=="male","Sex"]=1
    train_df.loc[train_df["Sex"] == "female", "Sex"] = 0
    train_df=train_df.astype({"Sex":float})
    #2
    #print(train_df.columns)
    #(train_df.dtypes)

    train_x=train_df.drop("Survived",axis=1).to_numpy()
    train_y = train_df["Survived"].to_numpy()

    #train_data=train_df.to_numpy()
    #heatmap=sns.heatmap(train_df.corr().abs(),annot=True)
    #plt.show()

    #print(train_x.shape)
    #print(train_y.shape)

    def create_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation="sigmoid"),#показала лучший результат
            #tf.keras.layers.Dense(2, activation='relu'),
        ])
    model=create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    #shutil.rmtree("logs\\fit")
    #time.sleep(5)
    time_stop=int(time.time())
    tensorboard=TensorBoard(log_dir="logs\\fit\\{}".format(time_stop))
    #tensorboard = TensorBoard(log_dir="logs\\fit\\NN1", update_freq='epoch')
    epochs = 240
    model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        validation_split=0.2,
        #batch_size=32,
        callbacks=[tensorboard]
    )
    model.save("models\\{}".format(time_stop))





