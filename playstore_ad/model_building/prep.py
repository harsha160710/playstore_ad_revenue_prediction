import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login,HfApi

api=HfApi(token=os.getenv('HF_TOKEN_1'))
DATASET_PATH="hf://datasets/Harsha1001/play-store-ad-revenue-prediction/playstore_revenue_analysis.csv"
df=pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

df.drop(columns=['app_id','app_name','updated_date'],inplace=True)
label_encoder=LabelEncoder()
df['app_category']=label_encoder.fit_tranform(df['app_category'])
df['free_or_paid']=label_encoder.fit_transform(df['free_or_paid'])
df['content_rating']=label_encoder.fit_transform(df['content_rating'])
df['screentime_category']=label_encoder.fit_transform(df['screentime_category'])

target_col='adv_revenue'

x=df.drop(columns=target_col)
y=df[target_col]

repo_id="Harsha1001/play-store-ad-revenue-prediction"
repo_type="dataset"

x_train,x_test,y_train,y_test=train_test_split(x,y,test_ration=0.2,random_state=42)

x_train.to_csv('x_train.csv',index=False)
x_test.to_csv('x_test.csv',index=False)
y_train.to_csv('y_train.csv',index=False)
y_test.to_csv('y_test.csv',index=False)

files=['x_train.csv','x_test.csv','y_train.csv','y_test.csv']

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path.split("/")[-1],
      repo_id=repo_id,
      repo_type=repo_type
      )
