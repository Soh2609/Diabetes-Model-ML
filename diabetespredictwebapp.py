import numpy as np
import pickle
import streamlit as st 
loaded_model = pickle.load(open("D:/Soham project work/trained_model.sav",'rb'))
#create function for predicting is person is diabetic
def diabetes_pred(input_data):
    
    input_data_as_nparray=np.asarray(input_data)
    input_data_reshaped=input_data_as_nparray.reshape(1,-1)
    predictn=loaded_model.predict(input_data_reshaped)
    print(predictn)
    if(predictn[0]==0):
      return"The person is Not diabetic"
    else:
      return"The person is diabetic"
    
def main():
    
    
    
    #Title for Webpage
    st.title("Multple diesease prediction model")
    
    #get input from user
    pregnencies = st.text_input("Number of pregnencies")
    glucose = st.text_input("glucose level")
    bloodpressure = st.text_input("Blood pressure value")
    SkinThickness = st.text_input("Skinthickness value")
    insulin = st.text_input("insulin level")
    BMI = st.text_input("BMI level")
    dbf = st.text_input("Diabetes pedigree function level")
    Age = st.text_input("Age ?")
    
    #Prediction
    diagnosis = ''
    
    #button for predict
    if st.button("Test result"):
        diagnosis = diabetes_pred([pregnencies,glucose,bloodpressure,SkinThickness,insulin,BMI,dbf,Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()