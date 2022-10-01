import tensorflow as tf
import pandas as pd
import numpy as np


def Convert_Categorical_to_Numeric(PandasSeries,interpolate=False):

    if (interpolate):
        #assert PandasSeries.isnull().values.any()==True,"Given column posses no nan values"
        PandasSeries.interpolate(inplace=True)
        PandasSeries.replace(PandasSeries.unique(),[i for i in range(PandasSeries.unique().shape[0])],inplace=True)
    else:
        assert PandasSeries.isnull().values.any()==False,"Given column posses some nan values, use INTERPOLATE flag"
        PandasSeries.replace(PandasSeries.unique(), [i for i in range(PandasSeries.unique().shape[0])], inplace=True)

    return PandasSeries



old_model=tf.keras.models.load_model("models\\1664585910")
test_df = pd.read_csv(r"D:\System_folders\Dowloads\titanic\test.csv")
gend_sub_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\gender_submission.csv")

submission_series=test_df["PassengerId"]

test_df.drop(["Name", "PassengerId", "Ticket"], axis=1, inplace=True)

test_df["Cabin"] = Convert_Categorical_to_Numeric(test_df["Cabin"], interpolate=True)
test_df["Embarked"] = Convert_Categorical_to_Numeric(test_df["Embarked"], interpolate=True)
test_df["Sex"] = Convert_Categorical_to_Numeric(test_df["Sex"])
test_df.interpolate(inplace=True)

predictions=(old_model.predict(test_df.to_numpy()))
#print(predictions)
survived=[]

for i in range(predictions.shape[0]):
    if (predictions[i]>0.5):
        #print("{0} IS GREATER THAN {1}".format(predictions[i],5.0))
        survived.append(1)
    else:
        #print("{0} IS LESS THAN {1}".format(predictions[i], 5.0))
        survived.append(0)

#print(gend_sub_df.head(10))
submission_table=submission_series.to_frame()

submission_table.insert(1,"Survived",survived,True)
#print(submission_table.head())

submission_table.to_csv("submission_table.csv",index=False)