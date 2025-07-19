import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn. metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pickle
df=pd.read_csv("C:\My Vs Code Projects\MyPythonProjects\AIMLproject\Recenttasks\Salary Data.csv")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
# remove nan values 
data=df.dropna()
print(data.shape)
# encoding of data
print(data.select_dtypes(include='number').corr()['Salary'])
le=LabelEncoder()
data['Genders']=le.fit_transform(data['Gender'])
print(data['Gender'].unique())
data.drop(columns=['Gender'],inplace=True)
print(data['Genders'])
# one hot encoder 
print(data.sample())
X=data.drop('Salary',axis=1)
y=data['Salary']
print(X)
print(y)
# train test split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=2)
# print(data['Job Title'])
oridinaal_cols=["Education Level"]
oridinaal_values=[["Bachelor's" ,"Master's" ,'PhD']]
onehotcols=['Job Title']
passthrough_cols=['Age','Genders','Years of Experience']
preprossor=ColumnTransformer(transformers=[
    ("edu",OrdinalEncoder(categories=oridinaal_values),oridinaal_cols),
    ("job",OneHotEncoder( handle_unknown='ignore',sparse_output=False,drop="first"),onehotcols)],remainder="passthrough")
pipe=Pipeline([
    ('preprocess',preprossor),
    ("model",LinearRegression())
])
pipe.fit(X_train,y_train)
print(X.info())
print(y)
user_input = pd.DataFrame([{
    'Age': 20,
    'Education Level': "Bachelor's",
    'Job Title': 'Software Engineer',
    'Years of Experience': 1,
    'Genders': 1
}])
print(user_input.info())
# Predict
predicted_salary = pipe.predict(X_test)
print("Predicted Salary:", predicted_salary[0])
print('r2 score',r2_score(y_test,predicted_salary))
# using random forset regrresor 
step1RFR=ColumnTransformer(transformers=[
    ('coln_trf',OrdinalEncoder(categories=oridinaal_values),oridinaal_cols),
    ('coln1_trf',OneHotEncoder(handle_unknown='ignore', sparse_output=False,drop='first'),onehotcols)
],remainder='passthrough')
step2RFR=RandomForestRegressor(n_estimators=100,
                                random_state=3,
                                max_samples=0.5,
                                max_features=0.75,
                                max_depth=15)
pipe1=Pipeline([
   ('step1',step1RFR),
   ('step2',step2RFR) 
])
pipe1.fit(X_train,y_train)
ypredRFR=pipe1.predict(X_test)
print(ypredRFR)
print("R score : ",r2_score(y_test,ypredRFR))
print("MAE RFR",mean_absolute_error(y_test,ypredRFR))
print(X_train.head())
# print(X_train.head())
# model dump 
pickle.dump(pipe1,open("SalModel.pkl",'wb'))
pickle.dump(data,open('saldf.pkl','wb'))