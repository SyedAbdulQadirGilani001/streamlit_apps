import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
header=st.container()
datasets=st.container()
features=st.container()
model_training=st.container()
with header:
    st.title('Machine Learning App')
    st.text('This is a machine learning app')
with datasets:
    st.header('Datasets')
    st.text('This is a datasets')
    df=sns.load_dataset('titanic')
    st.write(df.head())
    st.write(df.shape)
    st.bar_chart(df['class'].value_counts())
    st.subheader('Data Types')
    st.bar_chart(df.dtypes[1:].value_counts())
    st.subheader('Missing Values')
    st.bar_chart(df.isnull().sum()[1:].sample(5))
with features:
    st.header('Features')
    st.text('This is a features')
    st.markdown('**Categorical Features**')
    st.write(df.select_dtypes(include=['object']).columns)
    st.markdown('**Numerical Features**')
    st.write(df.select_dtypes(include=['int64','float64']).columns)
with model_training:
    st.header('Model Training')
    st.text('This is a model training')
    input,display=st.columns(2)
    input.slider('Age',min_value=0,max_value=100)
    max_depth=input.slider('Max Depth',min_value=1,max_value=10,value=5,step=1)
n_estimators=input.selectbox('Number of Estimators',options=[100,200,300,400,500,'No Input']) 
uinput_features=input.multiselect('Select Features',options=df.columns[1:])
display.subheader('Selected Features')
display.write(uinput_features)
display.subheader('Selected Max Depth')
display.write(max_depth)
display.subheader('Selected Number of Estimators')
display.write(n_estimators)
if st.button('Train Model'):
    X=df[uinput_features]
    y=df['survived']
    model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    model.fit(X,y)
    y_pred=model.predict(X)
    st.subheader('Model Performance')
    st.write('Mean Absolute Error:',mean_absolute_error(y,y_pred))
    st.write('Mean Squared Error:',mean_squared_error(y,y_pred))
    st.write('R2 Score:',r2_score(y,y_pred))
    st.subheader('Model Parameters')
    st.write(model.get_params())
