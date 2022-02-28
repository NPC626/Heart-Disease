
import streamlit as st 
import pandas as pd
import pickle 

st.write("""
# Heart Disease Prediction App
""")
st.write('---')

st.sidebar.header('Specify Input Parameters')

# X = pd.read_csv('heart.csv')

def user_input_features():
    Age = st.sidebar.slider('Age',0,130,value=25 )
    Sex  = st.radio('Sex',("Male","Female"))
    if Sex == "Male":
        Sex = 1
    elif Sex == "Female":
        Sex = 0
    Chest_Pain  = st.sidebar.slider('Chest Pain',1,4,value=2 )
    Resting_Blood_Pressure = st.sidebar.slider('Resting Blood Pressure',100,180,value=120 )
    Colestrol = st.sidebar.slider('Colestrol',150,300,value=250 )
    Fasting_Blood_Sugar = st.sidebar.slider('Fasting Blood Sugar',0,1,value=1 )
    Rest_ECG = st.sidebar.slider('Rest ECG',0,3,value=2 )
    MAX_Heart_Rate = st.sidebar.slider('Max Heart Rate',100,200,value=130 )
    Exercised_Induced_Angina = st.sidebar.slider('Exercised Induced Angina',0,1,value=1 )
    ST_Depression = st.text_input('ST Depression',key = "Depression" )
    slope  = st.sidebar.slider('Slope',0,5,value=2 )
    Major_Vessels = st.sidebar.slider('Major Vessels',0,10,value=2 )
    Thalessemia  = st.sidebar.slider('Thalessemia',0,10,value=7 )
    # Target
    data = {'Age': Age,
            'Sex': Sex,
            'Chest Pain': Chest_Pain,
            'Resting Blood Pressure': Resting_Blood_Pressure,
            'Colestrol': Colestrol,
            'Fasting Blood Sugar': Fasting_Blood_Sugar,
            'Rest ECG': Rest_ECG,
            'Max Heart Rate': MAX_Heart_Rate,
            'Exercised Induced Angina': Exercised_Induced_Angina,
            'Depression': ST_Depression,
            'Slope': slope,
            'Major Vessels': Major_Vessels,
            'Thalessemia': Thalessemia}
    features = pd.DataFrame(data, index=[0])
    
    return features

values = user_input_features()

st.header('Specified Input parameters')
st.write(values)
st.write('---')

# loaded_model = pickle.load(open(model.pkl, 'rb'))
# with open('model.pkl' , 'rb') as f:
#     lr = pickle.load(f)
pickled_model = pickle.load(open('model.pkl', 'rb'))

prediction = pickled_model.predict(values)

st.header('Prediction Result')
st.write(prediction)
st.write('---')


