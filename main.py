import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, Sequential


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    #assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


with tf.device ("/GPU:0"):
    train_df =pd.read_csv(r"D:\System_folders\Dowloads\titanic\train.csv")
    #test_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\test.csv")
    #gend_sub_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\gender_submission.csv")

    train_df.drop(["Name","Ticket","Cabin","Embarked","PassengerId","Age","SibSp","Parch"],axis=1,inplace=True)

    train_df.loc[train_df["Sex"]=="male","Sex"]=1
    train_df.loc[train_df["Sex"] == "female", "Sex"] = 0
    train_df=train_df.astype({"Sex":float})

    #print(train_df.columns)
    #(train_df.dtypes)

    train_ds,val_ds,test_ds=get_dataset_partitions_pd(train_df)

    print(train_ds.head(4))

    # train_x=train_ds.drop("Survived",axis=1).to_numpy()
    # train_y = train_ds["Survived"].to_numpy()
    #
    # val_x = val_ds.drop("Survived", axis=1).to_numpy()
    # val_y = val_ds["Survived"].to_numpy()

    #train_data=train_df.to_numpy()
    #heatmap=sns.heatmap(train_df.corr().abs(),annot=True)
    #plt.show()

    #print(train_x.shape)
    #print(train_y.shape)

    def create_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(27, activation='relu'),
            tf.keras.layers.Dense(11, activation='relu'),
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
        x=train_ds.drop("Survived",axis=1).to_numpy(),
        y=train_ds["Survived"].to_numpy(),
        epochs=epochs,
        validation_data=(val_ds.drop("Survived", axis=1).to_numpy(),val_ds["Survived"].to_numpy())
    )


#    SimpleNet.save("D:/model/oi")





