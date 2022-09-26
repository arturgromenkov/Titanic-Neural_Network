import tensorflow as tf
import pandas as pd
import numpy as np

old_model=tf.keras.models.load_model("models\\1664207523")
test_df = pd.read_csv(r"D:\System_folders\Dowloads\titanic\test.csv")
gend_sub_df=pd.read_csv(r"D:\System_folders\Dowloads\titanic\gender_submission.csv")
submission_series=test_df["PassengerId"]

#print(submission_table.head())

test_df.drop(["Name", "Ticket", "Cabin", "Embarked", "PassengerId", "Age", "SibSp", "Parch"], axis=1, inplace=True)
test_df.loc[test_df["Sex"] == "male", "Sex"] = 1
test_df.loc[test_df["Sex"] == "female", "Sex"] = 0
test_df = test_df.astype({"Sex": float})

predictions=(old_model.predict(test_df.to_numpy()))
survived=[]

#print(gend_sub_df.head(10))
for i in range(predictions.shape[0]):
    if predictions[i,0]<predictions[i,1]:
        survived.append(1)
    else:
        survived.append(0)

submission_table=submission_series.to_frame()

submission_table.insert(1,"Survived",survived,True)
print(submission_table.head())

submission_table.to_csv("submission_table.csv",index=False)