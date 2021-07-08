import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# df=pd.read_csv(r'C:\Users\Hotu Ram\OneDrive\Desktop\python\NB-classifier\Placement_Data_Full_Class.csv')
df=pd.read_csv('data\Placement_Data_Full_Class.csv')
df.head()

# pre- processing
df.isnull().sum()
df=df.drop(['salary'], axis = 1)
df.isnull().sum()
sns.countplot(x= 'status', data=df)
sns.countplot(x='status', hue='gender', data=df)
#convert categorical variable into indicator variables
workexp = pd.get_dummies(df['workex'])
workexp
# Concantenate data into original dataset.
df=pd.concat([df,workexp],axis=1)

# df.drop(['sl_no','gender','ssc_b','hsc_s','hsc_b','degree_t','workex','specialisation'],axis=1,inplace=True)
df=df.drop(['sl_no','gender','ssc_b','hsc_s','hsc_b','degree_t','workex','specialisation'],axis=1)
df
# Train Test Data
X=df.drop('status',axis=1)
y=df['status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
#Apply Naive Bayes theorem
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)

# compare actual and predicted
y_pred=model.predict(X_test)
df1=pd.DataFrame({'Actual Status':y_test,'Predicted Status':y_pred})
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)