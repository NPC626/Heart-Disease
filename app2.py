
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

X = pd.read_csv('A:\Madhana\Program\Python\Heart_disease\heart.csv')
X_train = X.drop("Target", axis=1)
y_train = X['Target']

# logisticRegr.fit(X_train, y_train)
xg_model = XGBClassifier()
xg_model.fit(X_train, y_train)

def user_input_features():
    Age = st.sidebar.slider('Age',0,130,value=25 )
    Sex  = st.radio('Sex',(1,0))
    # if Sex == "Male":
        # Sex = 1
    # elif Sex == "Female":
        # Sex = 0
    Chest_Pain  = st.sidebar.slider('Chest_Pain',1,4,value=2 )
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


