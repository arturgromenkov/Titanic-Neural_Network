import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, Sequential

class VisualisationCallback(tf.keras.callbacks.Callback):


    def __init__(self):
        self.train_loss=[]
        self.val_loss=[]
        self.epochs=[]
    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs["loss"])
        self.val_loss.append(logs[""])





with tf.device ("/GPU:0"):

    Name="NN1"
    visualisation=VisualisationCallback()

    #train_df =pd.read_csv(r"D:\System_folders\Dowloads\titanic\train.csv")
    train_df=pd.read_csv(r"C:\Users\Uzer\Downloads\train.csv")
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
            tf.keras.layers.Dense(27, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(11, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='relu'),
        ])
    model=create_model()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    epochs=30

    model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        validation_split=0.2,
        batch_size=32,
        callbacks=[visualisation]
    )


#    SimpleNet.save("D:/model/oi")





