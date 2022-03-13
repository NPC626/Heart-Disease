
import streamlit as st 
import pandas as pd
# import pickle 

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logisticRegr = LogisticRegression()

st.write("""
# Heart Disease Prediction App
""")
st.write('---')

st.sidebar.header('Specify Input Parameters')

X = pd.read_csv('heart.csv')
X_train = X.drop("Target", axis=1)
y_train = X['Target']

st.write("""Age: displays the age of the individual. \n
Sex: displays the gender of the individual using the following format :\n
    1 = male
    0 = female \n
Chest-pain type: displays the type of chest-pain experienced by the individual using the following format : \n
    1 = typical angina
    2 = atypical angina
    3 = non — anginal pain
    4 = asymptotic \n
Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit) \n
Serum Cholestrol: displays the serum cholesterol in mg/dl (unit) \n
Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl. \n
    If fasting blood sugar > 120mg/dl then : 1 (true)
    else : 0 (false) \n
Resting ECG : displays resting electrocardiographic results \n
    0 = normal
    1 = having ST-T wave abnormality
    2 = left ventricular hyperthrophy \n
Max heart rate achieved : displays the max heart rate achieved by an individual. \n
Exercise induced angina : \n
    1 = yes
    0 = no \n
ST depression induced by exercise relative to rest: displays the value which is an integer or float. \n
Peak exercise ST segment : \n
    1 = upsloping
    2 = flat
    3 = downsloping \n
Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float. \n
Thal : displays the thalassemia : \n
    3 = normal
    6 = fixed defect
    7 = reversible defect \n
Diagnosis of heart disease :Displays whether the individual is suffering from heart disease or not : \n
    0 = absence
    1, 2, 3, 4 = present. """)

# logisticRegr.fit(X_train, y_train)
xg_model = XGBClassifier()
xg_model.fit(X_train, y_train)

def user_input_features():
    Age = st.sidebar.slider('Age',0,130,value=25 )
    Sex  = st.radio('Sex',('Male','Female'))
    if Sex == "Male":
        Sex = 1
    elif Sex == "Female":
        Sex = 0
    Chest_Pain  = st.sidebar.slider('Chest_Pain',1,4,value=2 )
    # st.write('1 = typical angina | 2 = atypical angina | 3 = non — anginal pain | 4 = asymptotic')
    Resting_Blood_Pressure = st.sidebar.slider('Resting_Blood_Pressure',100,180,value=120 )
    Colestrol = st.sidebar.slider('Colestrol',150,300,value=250 )
    Fasting_Blood_Sugar = st.sidebar.slider('Fasting_Blood_Sugar',0,1,value=1 )
    Rest_ECG = st.sidebar.slider('Rest_ECG',0,3,value=2 )
    MAX_Heart_Rate = st.sidebar.slider('Max_Heart_Rate',100,200,value=130 )
    Exercised_Induced_Angina = st.sidebar.slider('Exercised_Induced_Angina',0,1,value=1 )
    ST_Depression = st.sidebar.slider('ST_Depression',0,5,value=2)
    Slope  = st.sidebar.slider('Slope',0,5,value=2 )
    Major_Vessels = st.sidebar.slider('Major_Vessels',0,10,value=2 )
    Thalessemia  = st.sidebar.slider('Thalessemia',0,10,value=7 )
    # Target
    data = {'Age': Age,
            'Sex': Sex,
            'Chest_Pain': Chest_Pain,
            'Resting_Blood_Pressure': Resting_Blood_Pressure,
            'Colestrol': Colestrol,
            'Fasting_Blood_Sugar': Fasting_Blood_Sugar,
            'Rest_ECG': Rest_ECG,
            'Max_Heart_Rate': MAX_Heart_Rate,
            'Exercised_Induced_Angina': Exercised_Induced_Angina,
            'ST_Depression': ST_Depression,
            'Slope': Slope,
            'Major_Vessels': Major_Vessels,
            'Thalessemia': Thalessemia}
    features = pd.DataFrame(data, index=[0])
    # features.astype(str)
    return features

values = user_input_features()
values.dtypes
st.header('Specified Input parameters')
st.write(values)
st.write('---')

# loaded_model = pickle.load(open(model.pkl, 'rb'))
# with open('model_pkl.pkl' , 'rb') as f:   #A:\Madhana\Program\Python\Heart_disease\
    # lr = pickle.load(f)
# prediction = logisticRegr.predict(values)
    
prediction = xg_model.predict(values)

st.header('Prediction Result')
st.write(prediction)
st.write('---')


