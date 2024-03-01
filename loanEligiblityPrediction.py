import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


df=pd.read_csv(r"C:\Users\ASUS\Downloads\Loan_Data1.csv")
df = df.dropna()
df
df.head()
df.info()
df["Gender"].value_counts()
df.replace({"Gender":{"Male":0,"Female":1}}, inplace=True)
df['Married'].value_counts()
df.replace({'Married':{'No':0,'Yes':1}},inplace=True)
df['Education'].value_counts()
df.replace({'Education':{'Not Graduate':0,'Graduate':1}},inplace=True)
df['Self_Employed'].value_counts()
df.replace({'Self_Employed':{'No':0,'Yes':1}},inplace=True)
df['Property_Area'].value_counts()
df.replace({'Property_Area':{'Rural':0,'Semiurban':2,'Urban':2}},inplace=True)
df['Loan_Status'].value_counts()
df.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)
df.head()
X=df.drop(['Loan_Status','Loan_ID'],axis=1)
X
y=df["Loan_Status"]
y
mn=MinMaxScaler()
X= mn.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=25)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
"""
data = pd.DataFrame({"Actual":y_test,
                    "Predict":y_pred})
data.shape
"""
plt.scatter(y_test,y_pred,color="pink")
plt.title("Actual VS Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid()
plt.show()
print (classification_report(y_test,y_pred))
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
X_new=df.sample(1)
X_new=X_new.drop("Loan_Status", axis=1)
X_new=X_new.drop("CoapplicantIncome", axis=1)
X_new
X_new=mn.fit_transform(X_new)
lr.predict(X_new)



