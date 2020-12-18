# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PIL import Image




# select the deasise
option = st.selectbox('what kind of disease you want to slect?',('Heart attack', 'Diabets', 'Covid 19','Breast cancer',))
st.write('You selected:', option)

    
# Create a title and sub-title

st.write("""
# heart attack detection""")


#open image

image = Image.open("C:/Users/ASUS/Desktop/big project/project 1 detectation des maladie/heart/heart.jpg")
st.image(image,use_column_width=True)

# video 
st.header('what is heart attack ?')
video_file = open('C:/Users/ASUS/Desktop/big project/project 1 detectation des maladie/heart/heart.mkv', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


#set a subbheader

st.header('data information')

# show data as a table

df = pd.read_csv("C:/Users/ASUS/Desktop/big project/project 1 detectation des maladie/heart/heart.csv")
st.dataframe(df.style.highlight_max(axis=0))

#show static on the data

st.write(df.describe())

# select best features
df=df.drop('fbs',axis=1)

# show data as chart
st.header('data plot')
st.line_chart(df)




# split data 
x=df.drop('target',axis=1)
y=df.target

#split data into 75% training and 25% test
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.25,random_state=0)


# get the feature imput from the users
def get_user_input():
    age = st.sidebar.slider('age',29,77,30)
    sex = st.sidebar.slider('sex = (1 = male; 0 = female)',0,1,0) # first = min / second = max / third = the cursor in page web
    cp = st.sidebar.slider('cp= chest pain type',0,3,0)
    trestbps = st.sidebar.slider('resting blood pressure (in mm Hg on admission to the hospital)',94,200,92)
    chol = st.sidebar.slider('serum cholestoral in mg/dl',126,564,126)
    restecg = st.sidebar.slider('resting electrocardiographic results',0.0,2.0,0.0)
    thalach = st.sidebar.slider('maximum heart rate achieved',71.0,202.0,71.0)
    exang = st.sidebar.slider('exercise induced angina (1 = yes; 0 = no)',0.0,1.0,0.0)
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest',0.0,6.2,0.0)
    slope = st.sidebar.slider('the slope of the peak exercise ST segment',0.0,2.0,0.0)
    ca = st.sidebar.slider('number of major vessels (0-3) colored by flourosopy',0.0,4.0,0.0)
    thal = st.sidebar.slider(' normal; 6 = fixed defect; 7 = reversable defect',0.0,3.0,0.0)
    
     # store a dictionarry into a variable
    user_data = {'age':age,
             'sex':sex,
             'cp= chest pain type':cp,
             'resting blood pressure (in mm Hg on admission to the hospital)':trestbps,
             'serum cholestoral in mg/dl':chol,
             'resting electrocardiographic results':restecg,
             'maximum heart rate achieved':thalach,
             'exercise induced angina (1 = yes; 0 = no)':exang,
             'ST depression induced by exercise relative to rest':oldpeak,
             'the slope of the peak exercise ST segment':slope,
             'number of major vessels (0-3) colored by flourosopy':ca,
             ' normal; 6 = fixed defect; 7 = reversable defect':thal
             }
#transform data in to data dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features
    
#store the user input into a data frame
user_input = get_user_input()

 #set a subheader and display the user_input
st.subheader('your data :')
st.write(user_input)

#create and train the model
model = LogisticRegression()
#Fit the model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#show the model metrics
st.subheader('accuracy_score:')
st.write( str(accuracy_score(y_test,model.predict(x_test))*100)+'%')
print("Accuracy --> ", model.score(x_test, y_test)*100)
    
#store the models prediction in a variable

prediction = model.predict(user_input)

# set a subheader and display the classification
st.subheader('resultat')
st.write(prediction)


# rating
st.write('Are you satisfied with the result')
if st.button('Yes'):
    st.write('thanks for you rate')
if st.button('No'):
    st.write('we will try to improvment')
else :
    st.write('So what is your rate :')



















